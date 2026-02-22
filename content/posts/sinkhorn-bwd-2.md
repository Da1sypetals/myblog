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

### 解决方法

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

---

## 算法

### 算法1：秩1修正（正定系统）

**1）准备右端项**
$$r = (G \odot R)\mathbf{1}, \quad c = (G \odot R)^T\mathbf{1}$$

**2）构建正定系统**
$$S = I - R^T R + \frac{1}{n}\mathbf{1}\mathbf{1}^T$$
**以及**
$$b = c - R^T r$$

**3）求解**
$$S \, \tilde{v} = b$$

**4）构造解**
$$u = r - R\tilde{v}$$
$$v = \tilde{v} - \left(\frac{1}{n}\mathbf{1}^T\tilde{v}\right)\mathbf{1}$$

**5）组装结果**
$$M_{ij} = u_i + v_j$$

**6）最终梯度**
$$\nabla_X L = (G - M) \odot R$$

### 算法2：不修正

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


---

## 实现

采取不修正的方式实现。

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

---

## 证明

### 秩1修正后，结果的正确性

**步骤1**：验证 $\tilde{v}$ 与 $v^*$ 的关系。

由 $S\tilde{v} = c - R^T r$ 和 $(I - R^T R)v^* = c - R^T r$，有：
$$(I - R^T R + \frac{1}{n}\mathbf{1}\mathbf{1}^T)\tilde{v} = (I - R^T R)v^*$$

即：
$$(I - R^T R)(\tilde{v} - v^*) = -\frac{1}{n}\mathbf{1}\mathbf{1}^T\tilde{v}$$

由于 $I - R^T R$ 的零空间由 $\mathbf{1}$ 张开（$R^T R \mathbf{1} = \mathbf{1}$），存在标量 $\alpha$ 使得：
$$\tilde{v} - v^* = -\frac{\mathbf{1}^T\tilde{v}}{n}\mathbf{1} + \alpha\mathbf{1}$$

取 $\alpha = \frac{\mathbf{1}^T\tilde{v}}{n} - \frac{\mathbf{1}^T v^*}{n}$，则：
$$\tilde{v} = v^* - \frac{\mathbf{1}^T v^*}{n}\mathbf{1} = v^* - \bar{v}^*\mathbf{1}$$

即 $\tilde{v}$ 是 $v^*$ 的去均值版本，满足 $\mathbf{1}^T\tilde{v} = 0$。

**步骤2**：计算 $u$。

$$u = r - R\tilde{v} = r - R(v^* - \bar{v}^*\mathbf{1}) = r - Rv^* + \bar{v}^*R\mathbf{1} = u^* + \bar{v}^*\mathbf{1}$$

其中利用了 $R\mathbf{1} = \mathbf{1}$。

**步骤3**：计算 $v$。

由于 $\mathbf{1}^T\tilde{v} = 0$，有：
$$v = \tilde{v} - 0 \cdot \mathbf{1} = \tilde{v} = v^* - \bar{v}^*\mathbf{1}$$

**步骤4**：验证 $M_{ij}$。

$$M_{ij} = u_i + v_j = (u^*_i + \bar{v}^*) + (v^*_j - \bar{v}^*) = u^*_i + v^*_j = M^*_{ij}$$

偏移量 $\bar{v}^*$ 在 $u_i$ 和 $v_j$ 中相互抵消，结果与原系统一致。$\square$

### 不修正

正确性的证明类似，而共轭梯度方法本身[可以解决半正定系统的求解](https://arxiv.org/pdf/1809.00793)。