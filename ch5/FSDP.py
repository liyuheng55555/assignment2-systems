import threading

import einops
import torch.nn
import torch.distributed as dist

'''
实现中，搞了forward前的收集，还剩这些：
1. forward之后的param恢复成sharded param
2. backward何时释放完整参数？
3. 异步all_gather?
'''
class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.param_infos: dict[str, dict] = {}
        self.param_names: dict[torch.nn.Parameter, str] = {}
        self.param_buffer: dict[torch.nn.Parameter, torch.Tensor] = {}
        self.compute_dtype = compute_dtype

        self.lock = threading.Lock()
        self.handles = []

        def pre_hook(m: torch.nn.Module, *args):
            for param in m.parameters():
                name = self.param_names[param]
                self.param_buffer[param] = param.data
                param.data = self._recover_param(name, param, expect_dtype=self.compute_dtype)

                # m.register_parameter(name, torch.nn.Parameter(full_tensor))

        def forward_hook(m: torch.nn.Module, *args):
            for param in m.parameters():
                param.data = self.param_buffer[param]
                # sub_module.register_parameter(param_name, torch.nn.Parameter(sharded_tensor))

        def all_reduce_hook(x: torch.nn.Parameter):
            handle = dist.all_reduce(x.grad, async_op=False)
            # with self.lock:
            #     self.handles.append(handle)
            x.data = self.param_buffer[x]
            # assert x.grad.dtype == self._sharded_param(x.grad).dtype
            # print(f"{x.data.dtype=} {x.grad.dtype=}  {self._sharded_param(x.grad).dtype=}")
            x.grad = self._sharded_param(x.grad).to(x.dtype)
            # if self.compute_dtype is not None:
            #     x.grad = x.grad.to(dtype=self.compute_dtype)
            x.grad /= dist.get_world_size()


        for param_name, param in module.named_parameters(recurse=True):
            sharded_tensor = self._sharded_param(param.data)
            self.param_infos[param_name] = {
                "shape": param.data.shape,
                "numel": param.numel(),
                "shard": sharded_tensor.numel()
            }
            param.data = sharded_tensor

            if param.requires_grad:
                param.register_post_accumulate_grad_hook(all_reduce_hook)

            self.param_names[param] = param_name

        for module_name, sub_module in module.named_children():
            sub_module.register_forward_pre_hook(pre_hook)
            sub_module.register_forward_hook(forward_hook)
            sub_module.register_full_backward_pre_hook(pre_hook)
            # sub_module.register_full_backward_hook(param_shard_hook)


    def _sharded_param(self, full_param: torch.Tensor) -> torch.Tensor:
        flatten = einops.rearrange(full_param, "... -> (...)")
        step = (flatten.numel() + dist.get_world_size()-1) // dist.get_world_size()
        start = dist.get_rank() * step
        result = flatten[start : min(start + step, flatten.numel())]
        if flatten.numel() < start + step:
            result = torch.nn.functional.pad(result, (0, start + step - flatten.numel()))
        return result


    def _recover_param(self, param_name: str, param: torch.nn.Parameter, expect_dtype) -> torch.Tensor:
        param_info = self.param_infos[param_name]
        buffer = [torch.zeros(param_info['shard'], device=param.device) for _ in range(dist.get_world_size())]
        dist.all_gather(buffer, param.data)
        flatten = torch.cat(buffer, dim=0)
        flatten = flatten[:param_info['numel']]
        full_tensor = flatten.reshape(param_info['shape'])
        full_tensor = full_tensor.to(dtype=expect_dtype)
        return full_tensor


    def get_full_params(self):
        result = {}
        for name, param in self.module.named_parameters():
            result[name] = self._recover_param(name, param, None)
        return result


    def forward(self, *inputs, **kwargs):
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