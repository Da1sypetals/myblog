+++
date = '2026-02-22T15:38:11+08:00'
title = '计算sinkhorn迭代的梯度 2：更简单的nxn系统'
+++

## 问题设定

> 注：$\odot$ 代表逐元素乘法。

1. **输入矩阵**：$X \in \mathbb{R}^{n \times n}$。
2. **指数化**：$P = \exp(X)$（逐元素指数）。
3. **Sinkhorn 结果**：得到双随机矩阵 $R = \text{diag}(u) P \text{diag}(v)$，其中 $u, v \in \mathbb{R}^n_{>0}$ 是缩放因子，满足：
   - 行和约束：$R \mathbf{1} = \mathbf{1} \implies u \odot (Pv) = \mathbf{1}$
   - 列和约束：$R^T \mathbf{1} = \mathbf{1} \implies v \odot (P^T u) = \mathbf{1}$
4. **损失函数**：$L = f(R)$，令 $G = \nabla_R L = \frac{\partial L}{\partial R}$ 为已知梯度。

## 求解

TLDR：通过对 Sinkhorn 的平衡条件进行隐函数求导，$\nabla_X L$ 的计算公式为：

$$\nabla_X L = (G - u \mathbf{1}^T - \mathbf{1} v^T) \odot R$$

其中 $u, v \in \mathbb{R}^n$ 是下列线性系统的解：

$$\begin{cases} u + R v = (G \odot R) \mathbf{1} \\ R^T u + v = (G \odot R)^T \mathbf{1} \end{cases}$$

### 矩阵形式

将上述方程改写成矩阵形式：

$$\begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} r \\ c \end{bmatrix}$$

其中 $r = (G \odot R)\mathbf{1}$，$c = (G \odot R)^T\mathbf{1}$。

记 $A = \begin{bmatrix} I & R \\ R^T & I \end{bmatrix}$，这是一个 $2n \times 2n$ 矩阵。

### 系统的奇异性

**定理**：$A$ 是奇异矩阵。

**证明**：考虑非零向量 $v = \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix}$。计算 $Av$：

$$Av = \begin{bmatrix} I\mathbf{1} - R\mathbf{1} \\ R^T\mathbf{1} - I\mathbf{1} \end{bmatrix} = \begin{bmatrix} \mathbf{1} - \mathbf{1} \\ \mathbf{1} - \mathbf{1} \end{bmatrix} = \mathbf{0}$$

由于存在非零向量在 $A$ 的零空间中，故 $\det(A) = 0$。$\square$

### 不变量

虽然解 $x = \begin{bmatrix} u \\ v \end{bmatrix}$ 包含不确定的偏移量（通解为 $x = x_0 + k \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix}$），但计算目标是确定的：

$$M = u\mathbf{1}^T + \mathbf{1}v^T \quad (\text{即 } M_{ij} = u_i + v_j)$$

**定理**：对于 $Ax=b$ 的任何解 $x$，矩阵 $M$ 是唯一确定的。

**证明**：将通解代入 $M$ 的表达式：

$$M(k) = (u_0 + k\mathbf{1})\mathbf{1}^T + \mathbf{1}(v_0 - k\mathbf{1})^T = u_0\mathbf{1}^T + k\mathbf{1}\mathbf{1}^T + \mathbf{1}v_0^T - k\mathbf{1}\mathbf{1}^T = u_0\mathbf{1}^T + \mathbf{1}v_0^T$$

$k$ 相关项相互抵消，$M$ 与具体解的选择无关。$\square$


## 算法

**1）准备右端项**
$$s_r = (G \odot R)\mathbf{1}, \quad s_c = (G \odot R)^T\mathbf{1}$$

**2）构建半正定系统**
$$S = I - R^T R$$
**以及**
$$b = s_c - R^T s_r$$

**3）用CG求解**
$$S \, x = b$$

**4）构造解**
$$u = r - Rx$$
$$v = x$$



**5）组装结果**
$$M_{ij} = u_i + v_j$$

**6）最终梯度**
$$\nabla_X L = (G - M) \odot R$$

可以证明这个方法和上述求解 $2n \times 2n$ 系统的方法数学上等价。CG[在一定条件下可以求解半正定系统](https://arxiv.org/pdf/1809.00793)，其余证明略。


## PyTorch 实现

```python
"""
Sinkhorn Backward Pass: n×n Rank-0 (Singular) System with Manual CG
Solves (I - R^T R) ṽ = b without rank-1 correction using manual conjugate gradient
"""

import torch

dtype = torch.float32
batch = 10001
n = 4
iters = 48

EPS = 1e-11

print(f"{n = }")
print(f"{iters = }")
print(f"{batch = }")


def sinkhorn_forward(M, iters=20):
    """Standard Sinkhorn forward pass"""
    P = torch.exp(M)
    R = P
    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)
    return R, P


def batch_cg_solve_singular(A, b):
    """
    Manual Conjugate Gradient solver for potentially singular systems.
    A: (batch, n, n) - system matrices
    b: (batch, n) - right hand side
    """
    batch_size, n, _ = A.shape
    device = A.device
    dtype = A.dtype

    # CG Initialization
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.einsum("bi,bi->b", r, r)

    # CG Iteration
    # Iteration count is n, which is theoretically guaranteed by CG algorithm
    for i in range(n):
        Ap = torch.einsum("bij,bj->bi", A, p)
        pAp = torch.einsum("bi,bi->b", p, Ap)
        alpha = rs_old / (pAp + EPS)
        x += torch.einsum("b,bi->bi", alpha, p)
        r -= torch.einsum("b,bi->bi", alpha, Ap)
        rs_new = torch.einsum("bi,bi->b", r, r)
        beta = rs_new / (rs_old + EPS)
        p = r + torch.einsum("b,bi->bi", beta, p)
        rs_old = rs_new

    return x


def sinkhorn_backward_n_rank0(grad_R, R, cg_iters=10):
    """
    Rank-0 method: Solve n×n singular system WITHOUT rank-1 correction
    Uses manual Conjugate Gradient to solve (I - R^T R) ṽ = b

    Algorithm steps:
    1. r = (G ⊙ R)1, c = (G ⊙ R)^T 1
    2. S0 = I - R^T R (SINGULAR), b = c - R^T r
    3. Solve: S0 ṽ = b using manual CG
    4. u = r - R ṽ, v = ṽ (CG naturally finds ~zero mean solution)
    5. M_{ij} = u_i + v_j
    6. ∇_X L = (G - M) ⊙ R
    """
    batch_size, n, _ = R.shape
    device = R.device
    dtype = R.dtype

    R_detached = R.detach()
    G = grad_R

    # Step 1: Prepare RHS
    r = (R_detached * G).sum(dim=-1)  # shape (batch, n)
    c = (R_detached * G).sum(dim=-2)  # shape (batch, n)

    # Step 2: Build n×n SINGULAR system (no rank-1 correction)
    # S0 = I - R^T R
    R_T = torch.einsum("bij->bji", R_detached)
    RTR = torch.einsum("bij,bjk->bik", R_T, R_detached)
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    S0 = eye - RTR  # SINGULAR matrix
    b = c - torch.einsum("bij,bj->bi", R_T, r)

    # Debug: Compute eigenvalues to verify singularity
    eigenvalues = torch.linalg.eigvalsh(S0)
    min_eig = eigenvalues.min(dim=-1).values
    max_eig = eigenvalues.max(dim=-1).values
    print("Rank-0 system eigenvalue statistics:")
    print(f"  Min eigenvalue:  {min_eig.min().item():.5f}")
    print(f"  Max eigenvalue:  {max_eig.max().item():.5f}")
    print(f"  Near-zero eigenvalues exist: {(eigenvalues.abs() < 1e-6).any().item()}")

    # Step 3: Solve S0 ṽ = b using manual CG
    v_tilde = batch_cg_solve_singular(S0, b, max_iter=cg_iters)

    # Step 4: Construct solution
    # CG naturally produces minimum-norm solution (~zero mean)
    u = r - torch.einsum("bij,bj->bi", R_detached, v_tilde)
    v = v_tilde

    # Step 5: Assemble M_{ij} = u_i + v_j
    M = u.unsqueeze(-1) + v.unsqueeze(-2)

    # Step 7: Final gradient
    grad_X = (G - M) * R_detached

    return grad_X


######################################################################
# Test Setup
######################################################################

# Generate random input
dist = torch.distributions.uniform.Uniform(0.0, 4.0)
M = dist.sample((batch, n, n))
M.requires_grad_()

# Forward pass (shared)
R, P = sinkhorn_forward(M, iters)
loss_weight = torch.randn_like(R)

######################################################################
# Method A: Autograd (Reference)
######################################################################
M.grad = None
loss_a = (R * loss_weight).sum()
loss_a.backward()
grad_M_autograd = M.grad.detach().clone()

######################################################################
# Method B: Rank-0 CG (Singular system, manual CG)
######################################################################
grad_R = loss_weight
grad_M_rank0_cg = sinkhorn_backward_n_rank0(grad_R, R, cg_iters=n)

######################################################################
# Comparison
######################################################################

g_ref = grad_M_autograd
g_rank0 = grad_M_rank0_cg

# Compute differences
abs_diff = (g_ref - g_rank0).abs()
rel_diff = abs_diff / (g_ref.abs() + 1e-12)

MAE = abs_diff.mean(dim=(-1, -2))
max_abs_diff = abs_diff.reshape(batch, -1).max(-1).values
mean_rel_diff = rel_diff.mean(dim=(-1, -2))
max_rel_diff = rel_diff.reshape(batch, -1).max(-1).values

print("\n" + "=" * 60)
print("GRADIENT COMPARISON: Autograd vs Rank-0 CG")
print("=" * 60)
print(f"Max MAE:           {MAE.max().item():.6e}")
print(f"Max max_abs_diff:  {max_abs_diff.max().item():.6e}")
print(f"Max mean_rel_diff: {mean_rel_diff.max().item():.6e}")
print(f"Max max_rel_diff:  {max_rel_diff.max().item():.6e}")

print("\n" + "=" * 60)
print("SAMPLE GRADIENTS (first batch)")
print("=" * 60)
print(f"\nAutograd reference:\n{g_ref[0]}")
print(f"\nRank-0 CG method:\n{g_rank0[0]}")
print(f"\nAbsolute difference:\n{abs_diff[0]}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

```

## Triton 实现

### 实现细节

在 Triton kernel 中，每个 thread block 处理一批（`tilesize` 个）独立的 $n \times n$ 系统。CG 的每次迭代需要计算 $S p = (I - R^T R) p$。

最直接的实现是在 kernel 入口处预计算 $R^T R$，然后在 CG 循环中复用：

```python
# 原始实现（tune_triton_new.py）
RTR = tl.dot(RT, R, input_precision="tf32")   # 预计算，常驻寄存器

for _ in range(n):
    Sp = p - tl.dot(RTR, p)   # 每次迭代 1 次 matvec
```


但是这样实现很慢：这里使用matmul的pattern和GEMM不同，只在开头计算一次，不足以填满 Tensor Core 的流水线，利用率低；实测比处理 $2n \times 2n$ 系统的原始实现还慢约 2 倍。

注意到 $S p$ 可以分解为两次连续的 matvec，而无需显式存储 $R^T R$：

$$S p = (I - R^T R) p = p - R^T \underbrace{(R p)}_{\text{中间结果}}$$

即先算 $q = Rp$（$n \times 1$），再算 $R^T q$（$n \times 1$），两次 matvec 的中间结果 $q$ 只需 $n$ 个寄存器，用完即释放。

对应的 Triton 实现：

```python
@triton.jit
def matvec_S(R, x):
    Rx = tl.dot(R, x,  input_precision="ieee")  # q = Rx
    RT = R.permute(0, 2, 1)
    RTRx = tl.dot(RT, Rx, input_precision="ieee")              # R^T q
    return x - RTRx
```

CG 循环中直接调用，不再持有 `RTR`：

```python
for _ in range(n_stream):
    Sp = matvec_S(R, p)
    pSp = tl.sum(p * Sp, ...)
    ...
```

与处理 $2n \times 2n$ 系统的baseline相比，在下列代码的设定下，可以加速1.4x。

### 代码

```python
from icecream import ic
import torch
import einops as ein
import triton
import triton.language as tl
from tqdm import trange
import time


# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


dtype = torch.float32
EPS = tl.constexpr(1e-10)


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


@triton.jit
def matvec_S(R, x):
    """
    S = I - R^T R, perform S @ x WITHOUT materializing RTR.
    Computes: x - R^T (R x)  using two matvecs.
    R: (tilesize, n, n)
    x: (tilesize, n, 1)
    returns: (tilesize, n, 1)
    """
    Rx = tl.dot(R, x, input_precision="ieee")           # (tilesize, n, 1)
    RT = R.permute(0, 2, 1)
    RTRx = tl.dot(RT, Rx, input_precision="ieee")       # (tilesize, n, 1)
    return x - RTRx


@triton.autotune(
    configs=[
        triton.Config({"tilesize": tilesize}, num_stages=1, num_warps=num_warps)
        for tilesize in [1, 2, 4, 8, 16, 32, 64]
        for num_warps in [1, 2, 4, 8]
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
    RT = R.permute(0, 2, 1)
    dR = dout_desc.load([seq_off, 0, 0])

    # Step 1: r = (G ⊙ R) 1,  c = (G ⊙ R)^T 1
    RdR = R * dR
    r = tl.sum(RdR, axis=-1).expand_dims(-1)   # (tilesize, n, 1)
    c = tl.sum(RdR, axis=-2).expand_dims(-1)   # (tilesize, n, 1)

    # Step 2: b = c - R^T r
    b = c - tl.dot(RT, r, input_precision="ieee")  # (tilesize, n, 1)

    # Step 3: CG to solve (I - R^T R) x = b
    # Key optimization: do NOT precompute RTR (avoids n² register pressure).
    # Instead, each matvec_S call does: x - R^T(Rx)  (two n×1 matvecs).
    x = tl.zeros((tilesize, n_stream, 1), dtype=tl.float32)
    res_cg = b - matvec_S(R, x)   # residual = b - S x = b (since x=0)
    p = res_cg
    r_normsq = tl.sum(res_cg * res_cg, axis=1, keep_dims=True)

    for _ in range(n_stream):
        Sp = matvec_S(R, p)
        pSp = tl.sum(p * Sp, axis=1, keep_dims=True)
        alpha = r_normsq / (pSp + EPS)

        x += alpha * p
        res_cg -= alpha * Sp

        r_new_normsq = tl.sum(res_cg * res_cg, axis=1, keep_dims=True)
        beta = r_new_normsq / (r_normsq + EPS)

        p = res_cg + beta * p
        r_normsq = r_new_normsq

    # Step 4: u = r - R x,  v = x
    u = r - tl.dot(R, x, input_precision="ieee")   # (tilesize, n, 1)
    v = x                                           # (tilesize, n, 1)

    # Step 5: M_ij = u_i + v_j  =>  M = u 1^T + 1 v^T
    # u: (tilesize, n, 1), v^T: (tilesize, 1, n)
    v_T = v.reshape(tilesize, 1, n_stream)
    M_mat = u + v_T   # broadcast -> (tilesize, n, n)

    # Step 6: grad = (G - M) ⊙ R
    res_tile = (dR - M_mat) * R

    res_desc.store([seq_off, 0, 0], res_tile)


def sinkhorn_bwd_implicit_cg(
    out: torch.Tensor,
    dout: torch.Tensor,
    repeat: int,
):
    seqlen = out.size(0)
    n_stream = out.size(1)
    ic(seqlen)
    ic(n_stream)

    res = torch.empty_like(out)

    def grid(META):  # META is the dict passed in @triton.autotune
        return (triton.cdiv(seqlen, META["tilesize"]), 1, 1)

    a = torch.randn(8192, 8192)
    for _ in trange(4):
        _ = a @ a
        # Compile the kernel by running warmup
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
        )
        torch.cuda.synchronize()

    # start
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(repeat):
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
        )

    # end
    torch.cuda.synchronize()
    end_event.record()

    elapsed_time_ms = start_event.elapsed_time(end_event)

    # Print timing results
    print(f"Kernel execution time ({repeat = }): {elapsed_time_ms:.3f} ms")
    print(f"Average time per iteration: {elapsed_time_ms / repeat:.3f} ms")

    return res


def main():
    seqlen = 65536
    n_stream = 16
    iters = 100
    repeat = 512
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
    # Method B: Implicit differentiation (n×n system, no RTR materialization)
    ######################################################################
    grad_R = loss_weight

    grad_M_implicit = sinkhorn_bwd_implicit_cg(R, grad_R, repeat=repeat)

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

    print(f"Max MAE = {max(MAE)}")
    print(f"Max max_abs_diff = {max(max_abs_diff)}")
    print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
    print(f"Max max_rel_diff = {max(max_rel_diff)}")

    print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
    print("\nGrad (implicit) sample:\n", g2[0, :3, :3])

    assert max(MAE) < 1e-7, f"Intolerable difference: MAE = {max(MAE)}"


if __name__ == "__main__":
    main()

```