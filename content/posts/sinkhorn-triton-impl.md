+++
date = '2026-01-28T20:05:25+08:00'
title = '备注：sinkhorn迭代的反向传播的triton实现'
+++

起因是希望用Tilelang实现一版，但是发现：

- TileLang的matmul不支持batched
- 尝试用reduce sum模拟matmul，结果获得上千行看不懂的编译器内部报错
- reduce也出现不明所以的layout报错

因此现用triton实现一版可用的，后续如果有闲再来尝试加速。

注：

- 必须用ieee精度，即使是tf32x3 精度也不够，很可能在cg迭代中累积了，会出现巨大偏差
- triton要求k>=16, 因此这里只能选取n_stream=16（deepseek论文中是4）
- 用dot的方式比sum的方式快，目前只是功能实现，暂时没有研究ir或者profile了解原因。


```python
from icecream import ic
import torch
import einops as ein
import triton
import triton.language as tl


# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


dtype = torch.float32

seqlen = 4096
tilesize = 32
n_stream = 16
iters = 100
print(f"{n_stream = }")
print(f"{iters = }")

EPS = tl.constexpr(1e-10)


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


@triton.jit
def matvec_A(R, x1, x2):
    """
    A = [I, R; R.T, I], perform A @ [x1, x2]
    R: (tilesize, n, n)
    x1/x2: (tilesize, n, 1)
    """
    Rx2 = tl.dot(R, x2, input_precision="ieee")
    RT = R.permute(0, 2, 1)
    RTx1 = tl.dot(RT, x1, input_precision="ieee")

    y1 = x1 + Rx2
    y2 = RTx1 + x2

    return y1, y2


# @triton.jit
# def matvec_A(R, x1, x2):
#     """
#     A = [I, R; R.T, I], perform A @ [x1, x2]
#     R: (tilesize, n, n)
#     x1/x2: (tilesize, n, 1)
#     """
#     x2_new = x2.permute(0, 2, 1)
#     Rx2 = tl.sum(R * x2_new, axis=-1, keep_dims=True)
#     RT = tl.permute(R, (0, 2, 1))
#     x1_new = x1.permute(0, 2, 1)
#     RTx1 = tl.sum(RT * x1_new, axis=-1, keep_dims=True)

#     y1 = x1 + Rx2
#     y2 = RTx1 + x2

#     return y1, y2


@triton.jit
def dot(a1, a2, b1, b2):
    """
    inputs: (tilesize, n, 1)
    returns: (tilesize, 1, 1)
    """
    sum1 = tl.sum(a1 * b1, axis=1, keep_dims=True)
    sum2 = tl.sum(a2 * b2, axis=1, keep_dims=True)

    return sum1 + sum2


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=4),
    ],
    key=[],
)
@triton.jit
def sinkhorn_bwd_implicit_cg_kernel(
    seqlen,
    out,
    dout,
    res,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    dout_stride_0,
    dout_stride_1,
    dout_stride_2,
    res_stride_0,
    res_stride_1,
    res_stride_2,
    n_stream: tl.constexpr,
    tilesize: tl.constexpr,
):
    out_desc = tl.make_tensor_descriptor(
        out,
        shape=[seqlen, n_stream, n_stream],
        strides=[out_stride_0, out_stride_1, out_stride_2],
        block_shape=[tilesize, n_stream, n_stream],
    )

    dout_desc = tl.make_tensor_descriptor(
        dout,
        shape=[seqlen, n_stream, n_stream],
        strides=[dout_stride_0, dout_stride_1, dout_stride_2],
        block_shape=[tilesize, n_stream, n_stream],
    )

    res_desc = tl.make_tensor_descriptor(
        res,
        shape=[seqlen, n_stream, n_stream],
        strides=[res_stride_0, res_stride_1, res_stride_2],
        block_shape=[tilesize, n_stream, n_stream],
    )

    seq_off = tl.program_id(0) * tilesize

    R = out_desc.load([seq_off, 0, 0])
    dR = dout_desc.load([seq_off, 0, 0])

    RdR = R * dR

    b1 = tl.sum(RdR, axis=-1).expand_dims(-1)
    b2 = tl.sum(RdR, axis=-2).expand_dims(-1)

    x1 = tl.zeros((tilesize, n_stream, 1), dtype=tl.float32)
    x2 = tl.zeros((tilesize, n_stream, 1), dtype=tl.float32)
    tmp1, tmp2 = matvec_A(R, x1, x2)
    r1 = b1 - tmp1
    r2 = b2 - tmp2
    p1, p2 = r1, r2
    r_normsq = dot(r1, r2, r1, r2)

    for _ in range(n_stream * 2):
        Ap1, Ap2 = matvec_A(R, p1, p2)
        pAp = dot(p1, p2, Ap1, Ap2)
        # VERY important to avoid divide by zero
        alpha = r_normsq / (pAp + EPS)

        x1 += alpha * p1
        x2 += alpha * p2
        r1 -= alpha * Ap1
        r2 -= alpha * Ap2

        r_new_normsq = dot(r1, r2, r1, r2)

        # not very important to avoid divide by zero, but it's good to have it
        beta = r_new_normsq / (r_normsq + EPS)

        p1 = r1 + beta * p1
        p2 = r2 + beta * p2

        r_normsq = r_new_normsq

    # x1: (tilesize, n_stream, 1)
    x2_expand = x2.reshape(tilesize, 1, n_stream)

    res_tile = dR - x1 - x2_expand
    res_tile *= R

    res_desc.store([seq_off, 0, 0], res_tile)


def sinkhorn_bwd_implicit_cg(
    out: torch.Tensor,
    dout: torch.Tensor,
    tilesize: int,
):
    seqlen = out.size(0)
    n_stream = out.size(1)

    res = torch.empty_like(out)

    def grid(META):
        return (triton.cdiv(seqlen, tilesize), 1, 1)

    sinkhorn_bwd_implicit_cg_kernel[grid](
        seqlen,
        out,
        dout,
        res,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        res.stride(0),
        res.stride(1),
        res.stride(2),
        n_stream,
        tilesize,
    )

    return res


######################################################################
# Variable
######################################################################
dist = torch.distributions.uniform.Uniform(0.0, 4.0)
device = torch.device("cuda")
M = dist.sample((seqlen, n_stream, n_stream)).to(device)
M.requires_grad_()


######################################################################
# Shared forward + one shared loss weight
######################################################################
R, P = sinkhorn_forward(M, iters)
loss_weight = torch.randn_like(R)

######################################################################
# Method A: Autograd
######################################################################
loss_a = (R * loss_weight).sum()
loss_a.backward()
grad_M_autograd = M.grad.detach().clone()

######################################################################
# Method B: Implicit differentiation
######################################################################
grad_R = loss_weight

grad_M_implicit = sinkhorn_bwd_implicit_cg(R, grad_R, tilesize=32)


######################################################################
# Compare
######################################################################
g1 = grad_M_autograd
g2 = grad_M_implicit

abs_diff = (g1 - g2).abs()
rel_diff = abs_diff / (g1.abs() + 1e-12)

print("Comparison of gradients dL/dM")
print("--------------------------------")


def format_list(ls):
    return [f"{x:.2e}" for x in ls]


MAE = abs_diff.mean(dim=(-1, -2)).tolist()
max_abs_diff = abs_diff.reshape(seqlen, -1).max(-1).values.tolist()
mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
max_rel_diff = rel_diff.reshape(seqlen, -1).max(-1).values.tolist()

# print(f"MAE: {format_list(MAE)}")
# print(f"max_abs_diff: {format_list(max_abs_diff)}")
# print(f"mean_rel_diff: {format_list(mean_rel_diff)}")
# print(f"max_rel_diff: {format_list(max_rel_diff)}")

print(f"Max MAE = {max(MAE)}")
print(f"Max max_abs_diff = {max(max_abs_diff)}")
print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
print(f"Max max_rel_diff = {max(max_rel_diff)}")

print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
print("\nGrad (implicit) sample:\n", g2[0, :3, :3])


```
