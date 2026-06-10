import threading
from dataclasses import dataclass

import einops
import torch.nn
import torch.distributed as dist
from torch.autograd.graph import saved_tensors_hooks

'''
实现中，搞了forward前的收集，还剩这些：
3. 异步all_gather?
'''


def report(tag="") -> (str, float):
    if dist.get_rank() != 0:
        return "", 0
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    # reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    # peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    torch.cuda.reset_max_memory_allocated()

    return (
        f"[{tag}] "
        f"allocated={allocated:.1f} MB, "
        # f"reserved={reserved:.1f} MB, "
        f"peak_allocated={peak_allocated:.1f} MB, "
        # f"peak_reserved={peak_reserved:.1f} MB"
    ), allocated


@dataclass
class ShardedTensor():
    id: int
    shard: torch.Tensor
    shape: torch.Size
    full_numel: int
    shard_numel: int
    dtype: torch.dtype



class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.sharded_param_infos = {}
        self.param_buffer: dict[torch.nn.Parameter, torch.Tensor] = {}
        self.param_ptrs = {}
        self.shard_ids = 0
        self.compute_dtype = compute_dtype
        self.shard_threshold = 100000
        self.MEMORY_REPORT = True

        self.lock = threading.Lock()
        self.handles = []

        def gather_hook(m: torch.nn.Module, *args):
            pre_report, pre_alloc = report(f"{m._get_name()} before gather hook")
            count = 0
            for name, param in m.named_parameters(recurse=False):
                if param in self.sharded_param_infos:
                    self.param_buffer[param] = param.data
                    param.data = self._recover_param(param, expect_dtype=self.compute_dtype)
                    self.param_ptrs[param.data.untyped_storage().data_ptr()] = 0
                    count += 1
            if self.MEMORY_REPORT and dist.get_rank() == 0 and count > 0:
                post_report, post_alloc = report(f"{m._get_name()} after gather hook")
                print(pre_report)
                print(post_report)
                print(f"diff = {post_alloc - pre_alloc:.2f} MB")

        def backward_gather_hook(m: torch.nn.Module, *args):

            gather_hook(m)

        def shard_hook(m: torch.nn.Module, *args):
            pre_report, pre_alloc = report(f"{m._get_name()} before shard hook")
            count = 0
            for name, param in m.named_parameters(recurse=False):
                if param in self.sharded_param_infos:
                    if param.data.untyped_storage().data_ptr() in self.param_ptrs:
                        del self.param_ptrs[param.data.untyped_storage().data_ptr()]
                    param.data = self.param_buffer[param]
                    del self.param_buffer[param]
                    count += 1
            if self.MEMORY_REPORT and dist.get_rank() == 0 and count > 0:
                post_report, post_alloc = report(f"{m._get_name()} after shard hook")
                print(pre_report)
                print(post_report)
                print(f"diff = {post_alloc - pre_alloc:.2f} MB")

        def sharded_all_reduce_hook(x: torch.nn.Parameter):
            handle = dist.all_reduce(x.grad, async_op=False)
            # with self.lock:
            #     self.handles.append(handle)
            if x.data.untyped_storage().data_ptr() in self.param_ptrs:
                del self.param_ptrs[x.data.untyped_storage().data_ptr()]
            x.data = self.param_buffer[x]
            x.grad = self._shard_param(x.grad).to(x.dtype)
            x.grad /= dist.get_world_size()

        def all_reduce_hook(x: torch.nn.Parameter):
            handle = dist.all_reduce(x.grad, async_op=False)
            x.grad /= dist.get_world_size()

        for param in module.parameters():
            if param.numel() >= self.shard_threshold:
                sharded_tensor = self._shard_param(param.data)
                self.sharded_param_infos[param] = {
                    "shape": param.data.shape,
                    "numel": param.numel(),
                    "shard": sharded_tensor.numel()
                }
                param.data = sharded_tensor
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(sharded_all_reduce_hook)
            else:
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(all_reduce_hook)

        for sub_module in module.modules():
            sub_module.register_forward_pre_hook(gather_hook)
            sub_module.register_forward_hook(shard_hook)
            sub_module.register_full_backward_pre_hook(gather_hook)


    def _shard_param(self, full_param: torch.Tensor) -> torch.Tensor:
        flatten = einops.rearrange(full_param, "... -> (...)")
        step = (flatten.numel() + dist.get_world_size()-1) // dist.get_world_size()
        start = dist.get_rank() * step
        result = flatten[start : min(start + step, flatten.numel())]
        if flatten.numel() < start + step:
            result = torch.nn.functional.pad(result, (0, start + step - flatten.numel()))
        return result


    def _recover_param(self, param: torch.nn.Parameter, expect_dtype) -> torch.Tensor:
        with torch.cuda.nvtx.range("recover_param"):
            param_info = self.sharded_param_infos[param]
            buffer = [torch.zeros(param_info['shard'], device=param.device) for _ in range(dist.get_world_size())]
            dist.all_gather(buffer, param.data)
            flatten = torch.cat(buffer, dim=0)
            flatten = flatten[:param_info['numel']]
            full_tensor = flatten.reshape(param_info['shape'])
            full_tensor = full_tensor.to(dtype=expect_dtype)
            return full_tensor

    def _recover_param_from_ShardedTensor(self, sharded_tensor: ShardedTensor):
        buffer = [torch.zeros(sharded_tensor.shard_numel, device=sharded_tensor.shard.device) for _ in range(dist.get_world_size())]
        # print(f"[{dist.get_rank()}] recover {sharded_tensor.id=} {sharded_tensor.shape=}")
        dist.all_gather(buffer, sharded_tensor.shard)
        flatten = torch.cat(buffer, dim=0)
        flatten = flatten[:sharded_tensor.full_numel]
        full_tensor = flatten.reshape(sharded_tensor.shape)
        full_tensor = full_tensor.to(dtype=sharded_tensor.dtype)
        return full_tensor

    def get_full_params(self):
        result = {}
        for name, param in self.module.named_parameters():
            if name.startswith("norm"):
                result[name] = param.data
            else:
                result[name] = self._recover_param(param, None)
        return result


    def pack_hook(self, t: torch.Tensor):
        if t.numel() < self.shard_threshold or t.untyped_storage().data_ptr() not in self.param_ptrs:
            return t
        # print("pack_hook")
        # self.param_ptrs.remove(t.untyped_storage().data_ptr())
        del self.param_ptrs[t.untyped_storage().data_ptr()]
        shard = self._shard_param(t)
        self.shard_ids += 1
        sharded_tensor = ShardedTensor(
            id=self.shard_ids,
            shard=shard,
            shape=t.shape,
            full_numel=t.numel(),
            shard_numel=shard.numel(),
            dtype=t.dtype,
        )
        # print(f"[{dist.get_rank()}] shard {sharded_tensor.id=} {sharded_tensor.shape=}")
        if sharded_tensor.shape == torch.Size([64, 256, 512]):
            print("shit!")
        return sharded_tensor


    def unpack_hook(self, t: torch.Tensor):
        if not isinstance(t, ShardedTensor):
            return t
        # print("unpack_hook")
        return self._recover_param_from_ShardedTensor(t)


    def forward(self, *inputs, **kwargs):
        with saved_tensors_hooks(pack_hook=self.pack_hook, unpack_hook=self.unpack_hook):
            return self.module.forward(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        return
        # for handle in self.handles:
        #     handle.wait()
        #     with self.lock:
        #         self.handles.clear()
        #     for parameter in self.module.parameters():
        #         if parameter.grad is not None:
        #             parameter.grad /= dist.get_world_size()
        #     return

    def finish_step(self):
        return
        # self.param_ptrs = {}
