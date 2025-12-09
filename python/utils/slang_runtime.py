import torch
import slangtorch


BLOCK_SIZE_1D = 32


def launch_1d(x, LEN):
    x.launchRaw(
        blockSize=(BLOCK_SIZE_1D, 1, 1),
        gridSize=(LEN // BLOCK_SIZE_1D + 1, 1, 1),
    )


def SlangShaderForwardGrad(x, d_x, p, d_p, impl_idx, force_sign, shader, ret_const, ret_impl):
    if impl_idx is None:
        impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
    r = torch.full_like(x[:,0], -1.0, requires_grad=True)
    impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
    out_idx = torch.zeros(x.shape[0], dtype=torch.int32)
    d_r = torch.zeros_like(r)
    d_impl_fn = torch.zeros_like(impl_fn)
    
    launch_1d(shader.run.fwd(
      x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
      ret_const=ret_const, ret_impl=ret_impl
    ), LEN=x.shape[0])

    return d_r, d_impl_fn
  
  
def SlangShaderForwardGradRGB(x, d_x, p, d_p, impl_idx, force_sign, shader, ret_const, ret_impl):
    if impl_idx is None:
        impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
    r = torch.full_like(x[:,0].unsqueeze(-1).repeat(1,3), -1.0, requires_grad=True)
    impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
    out_idx = torch.zeros(x.shape[0], dtype=torch.int32)
    d_r = torch.zeros_like(r)
    d_impl_fn = torch.zeros_like(impl_fn)
    
    launch_1d(shader.run.fwd(
      x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
      ret_const=ret_const, ret_impl=ret_impl
    ), LEN=x.shape[0])

    return d_r, d_impl_fn
  


  
class SlangShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, impl_idx, force_sign, shader, ret_const, ret_impl):
        if impl_idx is None:
            impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
        r = torch.full_like(x[:,0], -1.0, requires_grad=True)
        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        # r = torch.full_like(x[:,0], torch.inf, requires_grad=True)
        out_idx = torch.zeros(x.shape[0], dtype=torch.int32)

        launch_1d(shader.run(
            x=x, p=p, r=r, impl_fn=impl_fn, impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
            ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])
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

        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        
        if grad_output is not None:
            d_r = grad_output.contiguous()
        else:
            d_r = torch.zeros_like(r)
        
        if grad_impl_fn is not None:
            d_impl_fn = grad_impl_fn.contiguous()
        else:
            d_impl_fn = torch.zeros_like(impl_fn)
        
        # TODO: In backward pass for discontinuity, force_sign should be -1, ret_const should be False
        # and ret_impl should be True. maybe add assertions for this
        
        launch_1d(shader.run.bwd(
        x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
        ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])

        return d_x, d_p, None, None, None, None, None

class SlangShaderRGB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, impl_idx, force_sign, shader, ret_const, ret_impl):
        if impl_idx is None:
            impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
        r = torch.full_like(x[:,0].unsqueeze(-1).repeat(1,3), -1.0, requires_grad=True)
        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        # r = torch.full_like(x[:,0], torch.inf, requires_grad=True)
        out_idx = torch.zeros(x.shape[0], dtype=torch.int32)

        launch_1d(shader.run(
            x=x, p=p, r=r, impl_fn=impl_fn, impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
            ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])
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

        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        
        if grad_output is not None:
            d_r = grad_output.contiguous()
        else:
            d_r = torch.zeros_like(r)
        
        if grad_impl_fn is not None:
            d_impl_fn = grad_impl_fn.contiguous()
        else:
            d_impl_fn = torch.zeros_like(impl_fn)
        
        # TODO: In backward pass for discontinuity, force_sign should be -1, ret_const should be False
        # and ret_impl should be True. maybe add assertions for this
        
        launch_1d(shader.run.bwd(
        x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
        ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])

        return d_x, d_p, None, None, None, None, None


