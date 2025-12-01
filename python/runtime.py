import torch
import slangtorch

BLOCK_SIZE_1D = 32


def _launch_1d(kernel, length: int):
    kernel.launchRaw(
        blockSize=(BLOCK_SIZE_1D, 1, 1),
        gridSize=(length // BLOCK_SIZE_1D + 1, 1, 1),
    )


class SlangShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, impl_idx, force_sign, shader, ret_const, ret_impl):
        if impl_idx is None:
            impl_idx = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device) - 1
        r = torch.full_like(x[:, 0], -1.0, requires_grad=True)
        impl_fn = torch.full_like(x[:, 0], 1000.0, requires_grad=True)
        out_idx = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)

        _launch_1d(
            shader.run(
                x=x,
                p=p,
                r=r,
                impl_fn=impl_fn,
                impl_idx=impl_idx,
                out_idx=out_idx,
                force_sign=force_sign,
                ret_const=ret_const,
                ret_impl=ret_impl,
            ),
            length=x.shape[0],
        )
        ctx.save_for_backward(x, p, r, impl_fn, impl_idx, out_idx)
        ctx.shader = shader
        ctx.ret_const = ret_const
        ctx.ret_impl = ret_impl
        ctx.force_sign = force_sign
        return r, impl_fn, impl_idx, out_idx

    @staticmethod
    def backward(ctx, grad_output=None, grad_impl_fn=None, grad_impl_idx=None, grad_out_idx=None):
        x, p, r, impl_fn, impl_idx, out_idx = ctx.saved_tensors
        shader = ctx.shader
        ret_const = ctx.ret_const
        ret_impl = ctx.ret_impl
        force_sign = ctx.force_sign
        d_x = torch.zeros_like(x)
        d_p = torch.zeros_like(p)

        impl_fn = torch.full_like(x[:, 0], 1000.0, requires_grad=True)

        d_r = grad_output.contiguous() if grad_output is not None else torch.zeros_like(r)

        d_impl_fn = torch.zeros_like(impl_fn)

        # For bwd, slangtorch expects DiffTensorView args as (primal, adjoint) tuples.
        launch_obj = shader.run.bwd(
            x=(x, d_x),
            p=(p, d_p),
            r=(r, d_r),
            impl_fn=(impl_fn, d_impl_fn),
            impl_idx=impl_idx,
            out_idx=out_idx,
            force_sign=force_sign,
            ret_const=ret_const,
            ret_impl=ret_impl,
        )
        _launch_1d(launch_obj, length=x.shape[0])

        return d_x, d_p, None, None, None, None, None


class BaseIntegrandSlang(torch.nn.Module):
    def __init__(self, shader_path, params):
        super().__init__()
        self.shader = slangtorch.loadModule(str(shader_path))
        self.p = torch.nn.Parameter(params)

    def forward(self, x, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        r, impl_fn, impl_idx, out_idx = SlangShader.apply(
            x, self.p, impl_idx, force_sign, self.shader, ret_const, ret_impl
        )
        if ret_const:
            return out_idx
        if ret_impl:
            return impl_fn, impl_idx
        return r
