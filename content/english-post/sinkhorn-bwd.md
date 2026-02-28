+++
date = '2026-02-28T14:40:55+08:00'
draft = true
title = 'Computing Sinkhorn Iteration Gradients Without Reverting the Forward Iterations'
+++

_Translated by Kimi K2.5_

## Problem Setup

### Problem Statement

> Note: $\odot$ denotes element-wise multiplication.

1. Input matrix: $X \in \mathbb{R}^{n \times n}$.

2. $P = \exp(X)$ (element-wise).
3. Through Sinkhorn-Knopp iteration on $P$, we obtain bistochastic matrix $R = \text{diag}(\alpha) P \text{diag}(\beta)$, where $\alpha, \beta \in \mathbb{R}^n_{>0}$ are scaling factors satisfying:
    - Row sum constraint: $R \mathbf{1} = \mathbf{1} \implies \alpha \odot (P\beta) = \mathbf{1}$
    - Column sum constraint: $R^T \mathbf{1} = \mathbf{1} \implies \beta \odot (P^T \alpha) = \mathbf{1}$
4. Loss function: $L = f(R)$, let $G = \nabla_R L = \frac{\partial L}{\partial R}$ be the known gradient.

### Objective

Gradient of $L$ with respect to $X$: $\frac{\partial L}{\partial X}$.

### TLDR

By solving the following system using the CG method:

$$\begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} (G \odot R) \mathbf{1} \\ (G \odot R)^T \mathbf{1} \end{bmatrix}$$

we obtain the gradient of $L$ with respect to $X$:
$$\nabla_X L = (G - u \mathbf{1}^T - \mathbf{1} v^T) \odot R$$

This method converges when the forward Sinkhorn-Knopp iteration is sufficiently converged.


## Derivation

Our goal is to compute $\frac{\partial L}{\partial X}$. By the chain rule:


$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial R} \cdot \frac{\partial R}{\partial P} \cdot \frac{\partial P}{\partial X}$$

Since $P_{ij} = e^{X_{ij}} \implies \frac{\partial P_{ij}}{\partial X_{ij}} = P_{ij}$, if we can find $\frac{\partial L}{\partial P}$, the final result is $\nabla_X L = \nabla_P L \odot P$.

Through implicit differentiation of the Sinkhorn balancing conditions, we can prove the following formula for $\nabla_X L$ (derivation omitted):

$$\nabla_X L = (G - u \mathbf{1}^T - \mathbf{1} v^T) \odot R$$

where $u, v \in \mathbb{R}^n$ are solutions to the following linear system, with the right-hand sides being the *row sums* and *column sums* of $G \odot R$ respectively:

$$\begin{cases} u + R v = (G \odot R) \mathbf{1} \\ R^T u + v = (G \odot R)^T \mathbf{1} \end{cases}$$

### Solving the Linear System

Rewriting the above equations in matrix form:

$$\begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} (G \odot R) \mathbf{1} \\ (G \odot R)^T \mathbf{1} \end{bmatrix} = b_0$$

Solve this linear system to obtain $u$ and $v$.

### Assembling the Gradient

With $u$ and $v$ obtained, substitute into:

$$\frac{\partial L}{\partial X_{ij}} = (G_{ij} - u_i - v_j) R_{ij} = (G_{ij} - (u_i + v_j)) R_{ij}$$

For each $i,j$, what we need to solve from the above equation is $u_i + v_j$.


### Properties

Let $A=\begin{bmatrix} I & R \\ R^T & I \end{bmatrix}$.

#### 1. Multiple Solutions
Proof:

Consider the non-zero vector $w = \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix}$ (where $\mathbf{1}$ is the $n$-dimensional all-ones column vector).
Compute $Aw$:

$$Aw = \begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = \begin{bmatrix} I\mathbf{1} - R\mathbf{1} \\ R^T\mathbf{1} - I\mathbf{1} \end{bmatrix}$$

By the bistochastic matrix properties $R\mathbf{1} = \mathbf{1}$ and $R^T\mathbf{1} = \mathbf{1}$:

$$Aw = \begin{bmatrix} \mathbf{1} - \mathbf{1} \\ \mathbf{1} - \mathbf{1} \end{bmatrix} = \mathbf{0}$$

Since there exists a non-zero vector in the null space of $A$, $\det(A) = 0$.

Intuition: The row and column sums of a bistochastic matrix have redundancy (e.g., if row sums equal 1, each row really only needs its first $n-1$ elements).

#### 2. Invariants

##### 1) Solution Space of Linear System $Ax = b$
Since $A$ is singular, for a given vector $b$, if the equation has a solution, it must have infinitely many. The general solution form is:

$$x = \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} u_0 \\ v_0 \end{bmatrix} + k \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = \begin{bmatrix} u_0 + k\mathbf{1} \\ v_0 - k\mathbf{1} \end{bmatrix}$$

where $\begin{bmatrix} u_0 \\ v_0 \end{bmatrix}$ is a particular solution and $k$ is an arbitrary real scalar.


##### 2) Invariant
Although the solution $x$ contains an uncertain offset $k$, our computational target is determinate.
Our computational target is matrix $M$, defined as:

$$M = u\mathbf{1}^T + \mathbf{1}v^T \quad (\text{i.e., } M_{ij} = u_i + v_j)$$
Proof of uniqueness:
Substitute the general solution with free variable $k$ into the expression for $M$:

$$M(k) = (u_0 + k\mathbf{1})\mathbf{1}^T + \mathbf{1}(v_0 - k\mathbf{1})^T$$

Expand using matrix distributivity:

$$M(k) = u_0\mathbf{1}^T + k(\mathbf{1}\mathbf{1}^T) + \mathbf{1}v_0^T - k(\mathbf{1}\mathbf{1}^T)$$

Cancel the $k$-related terms:

$$M(k) = u_0\mathbf{1}^T + \mathbf{1}v_0^T = M_{\text{fixed}}$$
Conclusion:
For any solution $x$ of $Ax=b$, the matrix $M$ computed from them is determinate. That is:

$$M = f(R, b)$$

$M$ is a determinate function of $R$ and $b$, unaffected by the specific choice of solution; therefore, as long as the solver converges, the correct gradient can be computed.

#### 3. Transformation of Form

Eliminating variables from the original system:
$$R^T(s_r - Rv) + v = s_c \implies (I - R^T R)v = s_c - R^T s_r$$

We obtain the new equation $S\tilde{v} = b$, where $S = I - R^T R$, $b = s_c - R^T s_r$, and $\tilde{v}$ is the solution to the new system.

Here, $S$ is symmetric positive semidefinite:

**1) Symmetry**
$$S^T = I - (R^T R)^T = I - R^T R = S \quad \checkmark$$

**2) Positive Semidefiniteness**

For any $x \in \mathbb{R}^n$:
$$x^T S x = \|x\|_2^2 - \|Rx\|_2^2$$

We only need to prove $\|Rx\|_2 \leq \|x\|_2$.

By row stochasticity ($\sum_j R_{ij} = 1$) and Jensen's inequality:
$$\left(\sum_j R_{ij} x_j\right)^2 \leq \sum_j R_{ij} x_j^2$$

Summing over $i$:
$$\|Rx\|_2^2 = \sum_i \left(\sum_j R_{ij} x_j\right)^2 \leq \sum_i \sum_j R_{ij} x_j^2 = \sum_j x_j^2 \sum_i R_{ij} = \|x\|_2^2$$

Thus $x^T S x \geq 0$.

Since

$$S\mathbf{1} = \mathbf{1} - R^T(R\mathbf{1}) = \mathbf{1} - R^T\mathbf{1} = \mathbf{0}$$

$S$ is not positive definite.


## Algorithm

**1) Prepare right-hand side**
$$s_r = (G \odot R)\mathbf{1}, \quad s_c = (G \odot R)^T\mathbf{1}$$

**2) Build positive semidefinite system**
$$S = I - R^T R$$
**and**
$$b = s_c - R^T s_r$$

**3) Solve with CG**
$$S \, \tilde{v} = b$$

**4) Construct solution**
$$u = s_r - R\tilde{v}$$
$$v = \tilde{v}$$



**5) Assemble result**
$$M_{ij} = u_i + v_j$$

**6) Final gradient**
$$\nabla_X L = (G - M) \odot R$$

CG [can solve positive semidefinite systems under certain conditions](https://arxiv.org/pdf/1809.00793).


## PyTorch Implementation

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

## Triton Implementation

### Implementation Details

In the Triton kernel, each thread block processes a batch of (`tilesize`) independent $n \times n$ systems. Each iteration of CG requires computing $S p = (I - R^T R) p$.

The most straightforward implementation is to precompute $R^T R$ at the kernel entry and reuse it in the CG loop:

```python
# Original implementation (tune_triton_new.py)
RTR = tl.dot(RT, R, input_precision="tf32")   # Precompute

for _ in range(n):
    Sp = p - tl.dot(RTR, p)   # 1 matvec per iteration
```


However, this implementation is slow: the matmul pattern here differs from GEMM, and computing only once at the beginning is insufficient to fill the Tensor Core pipeline, resulting in low utilization; in practice, it's about 2× slower than the original implementation handling $2n \times 2n$ systems.

Notice that $S p$ can be decomposed into two consecutive matvecs without explicitly storing $R^T R$:

$$S p = (I - R^T R) p = p - R^T \underbrace{(R p)}_{\text{intermediate result}}$$

That is, first compute $q = Rp$ ($n \times 1$), then compute $R^T q$ ($n \times 1$).

The corresponding Triton implementation:

```python
@triton.jit
def matvec_S(R, x):
    Rx = tl.dot(R, x,  input_precision="ieee")  # q = Rx
    RT = R.permute(0, 2, 1)
    RTRx = tl.dot(RT, Rx, input_precision="ieee")              # R^T q
    return x - RTRx
```

Call directly in the CG loop without holding `RTR`:

```python
for _ in range(n_stream):
    Sp = matvec_S(R, p)
    pSp = tl.sum(p * Sp, ...)
    ...
```

Compared to the baseline handling $2n \times 2n$ systems, with the settings in the following code, this achieves a 1.4× speedup.

### Code

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

    # Step 1: s_r = (G ⊙ R) 1,  s_c = (G ⊙ R)^T 1
    RdR = R * dR
    s_r = tl.sum(RdR, axis=-1).expand_dims(-1)   # (tilesize, n, 1)
    s_c = tl.sum(RdR, axis=-2).expand_dims(-1)   # (tilesize, n, 1)

    # Step 2: b = s_c - R^T s_r
    b = s_c - tl.dot(RT, s_r, input_precision="ieee")  # (tilesize, n, 1)

    # Step 3: CG to solve (I - R^T R) x = b
    # Key optimization: do NOT precompute RTR
    # Instead, each matvec_S call does: x - R^T(Rx)  (two n×1 matvecs).
    x = tl.zeros((tilesize, n_stream, 1), dtype=tl.float32)
    r = b - matvec_S(R, x)   # residual = b - S x = b (since x=0)
    p = r
    r_normsq = tl.sum(r * r, axis=1, keep_dims=True)

    for _ in range(n_stream):
        Sp = matvec_S(R, p)
        pSp = tl.sum(p * Sp, axis=1, keep_dims=True)
        alpha = r_normsq / (pSp + EPS)

        x += alpha * p
        r -= alpha * Sp

        r_new_normsq = tl.sum(r * r, axis=1, keep_dims=True)
        beta = r_new_normsq / (r_normsq + EPS)

        p = r + beta * p
        r_normsq = r_new_normsq

    # Step 4: u = s_r - R x,  v = x
    u = s_r - tl.dot(R, x, input_precision="ieee")   # (tilesize, n, 1)
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