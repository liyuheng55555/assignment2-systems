import einops
import torch.nn
import torch.distributed as dist

'''
实现中，搞了forward前的收集，还剩这些：
1. forward之后的param恢复成sharded param
2. backward怎么搞
3. 异步all_gather
'''
class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.param_infos: dict[torch.nn.Module, dict[str, dict]] = {}

        def forward_pre_hook(m: torch.nn.Module, inputs):
            for name, sharded_param in m.named_parameters():
                param_info = self.param_infos[m][name]
                buffer = [torch.zeros(param_info['shard'], device=sharded_param.device) for _ in range(dist.get_world_size())]
                dist.all_gather(buffer, sharded_param.data)
                flatten = torch.cat(buffer, dim=0)
                flatten = flatten[:param_info['numel']]
                full_tensor = flatten.reshape(param_info['shape'])
                m.register_parameter(name, torch.nn.Parameter(full_tensor))

        for module_name, sub_module in module.named_children():
            self.param_infos[sub_module] = {}
            for param_name, param in sub_module.named_parameters(recurse=False):
                sharded_tensor = self._sharded_param(param.data)
                sub_module.register_parameter(param_name, torch.nn.Parameter(sharded_tensor))
                self.param_infos[sub_module][param_name] = {
                    "shape": param.data.shape,
                    "numel": param.numel(),
                    "shard": sharded_tensor.numel()
                }
            sub_module.register_forward_pre_hook(forward_pre_hook)


    def _sharded_param(self, full_param: torch.Tensor) -> torch.Tensor:
        flatten = einops.rearrange(full_param, "... -> (...)")
        step = (flatten.numel() + dist.get_world_size()-1) // dist.get_world_size()
        start = dist.get_rank() * step
        result = flatten[start : min(start + step, flatten.numel())]
        if flatten.numel() < start + step:
            result = torch.nn.functional.pad(result, (0, start + step - flatten.numel()))
        return result


    def _build_full_name(self, module_name: str, param_name: torch.nn.Parameter):
        pass



    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        pass