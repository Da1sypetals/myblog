+++
date = '2026-01-05T22:07:08+08:00'
title = '不通过反转正向传播的方式计算sinkhorn迭代的梯度'
+++


## 问题设定

### 问题

> 注：$\odot$ 代表逐元素乘法。

1. 输入矩阵: $X \in \mathbb{R}^{n \times n}$。

2. $P = \exp(X)$（element-wise）。
3. 通过对 $P$ 进行 Sinkhorn-knopp迭代，得到bistochastic matrix $R = \text{diag}(\alpha) P \text{diag}(\beta)$，其中 $\alpha, \beta \in \mathbb{R}^n_{>0}$ 是缩放因子，满足：
    - 行和约束：$R \mathbf{1} = \mathbf{1} \implies \alpha \odot (P\beta) = \mathbf{1}$
    - 列和约束：$R^T \mathbf{1} = \mathbf{1} \implies \beta \odot (P^T \alpha) = \mathbf{1}$
4. 损失函数: $L = f(R)$，令 $G = \nabla_R L = \frac{\partial L}{\partial R}$ 为已知梯度。

### 目标

$L$ 对 $X$ 的梯度：$\frac{\partial L}{\partial X}$。

### TLDR

通过使用CG方法求解下列方程：

$$\begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} (G \odot R) \mathbf{1} \\ (G \odot R)^T \mathbf{1} \end{bmatrix}$$

可以得到 $L$ 对 $X$ 的梯度：
$$\nabla_X L = (G - u \mathbf{1}^T - \mathbf{1} v^T) \odot R$$

在前向sinkhorn-knopp迭代充分收敛的条件下，该方法能够收敛。


## 求解
我们的目标是求 $\frac{\partial L}{\partial X}$。根据链式法则：


$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial R} \cdot \frac{\partial R}{\partial P} \cdot \frac{\partial P}{\partial X}$$

由于 $P_{ij} = e^{X_{ij}} \implies \frac{\partial P_{ij}}{\partial X_{ij}} = P_{ij}$, 若能求出 $\frac{\partial L}{\partial P}$，最终结果就是 $\nabla_X L = \nabla_P L \odot P$。

通过对 Sinkhorn 的平衡条件进行隐函数求导，可以证明得到 $\nabla_X L$ 的计算公式如下(过程略......):

$$\nabla_X L = (G - u \mathbf{1}^T - \mathbf{1} v^T) \odot R$$

其中 $u, v \in \mathbb{R}^n$ 是下列线性系统的解, 等号右边分别是$G \odot R$ 的*行和*和*列和*：

$$\begin{cases} u + R v = (G \odot R) \mathbf{1} \\ R^T u + v = (G \odot R)^T \mathbf{1} \end{cases}$$

## 具体实现步骤
### 1）求解线性系统

将上述方程改写成矩阵形式：

$$\begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} (G \odot R) \mathbf{1} \\ (G \odot R)^T \mathbf{1} \end{bmatrix} = b$$

求解上述线性系统，得到 $u$ 和 $v$。

### 2）组装梯度

得到 $u$ 和 $v$ 后，代入：

$$\frac{\partial L}{\partial X_{ij}} = (G_{ij} - u_i - v_j) R_{ij} = (G_{ij} - (u_i + v_j)) R_{ij}$$

对于每一个$i,j$，我们需要从上述方程中解得的就是$u_i + v_j$。


## 问题

记$A=\begin{bmatrix} I & R \\ R^T & I \end{bmatrix}$。

### 1. 多解
证明：

考虑非零向量 $w = \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix}$（其中 $\mathbf{1}$ 为全 1 的 $n$ 维列向量）。
计算 $Aw$：

$$Aw = \begin{bmatrix} I & R \\ R^T & I \end{bmatrix} \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = \begin{bmatrix} I\mathbf{1} - R\mathbf{1} \\ R^T\mathbf{1} - I\mathbf{1} \end{bmatrix}$$

根据bistochastic matrix性质 $R\mathbf{1} = \mathbf{1}$ 和 $R^T\mathbf{1} = \mathbf{1}$：

$$Aw = \begin{bmatrix} \mathbf{1} - \mathbf{1} \\ \mathbf{1} - \mathbf{1} \end{bmatrix} = \mathbf{0}$$

由于存在非零向量在 $A$ 的零空间（Null space）中，故 $\det(A) = 0$。

直观理解：bistochastic matrix的行和列和存在冗余（例如，行和为1，因此每行其实只需知道前n-1个元素）。

### 2. 不变量

#### 1) 线性方程组 $Ax = b$ 的解空间
由于 $A$ 是奇异的，对于给定的向量 $b$，如果方程有解，则必有无穷多解。其通解形式为：

$$x = \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} u_0 \\ v_0 \end{bmatrix} + k \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = \begin{bmatrix} u_0 + k\mathbf{1} \\ v_0 - k\mathbf{1} \end{bmatrix}$$

其中 $\begin{bmatrix} u_0 \\ v_0 \end{bmatrix}$ 是一个特解，$k$ 是任意实数标量。


#### 2) 不变量
虽然解 $x$ 包含不确定的偏移量 $k$，但我们的计算目标是确定的。
我们的计算目标是矩阵 $M$，定义为：

$$M = u\mathbf{1}^T + \mathbf{1}v^T \quad (\text{即 } M_{ij} = u_i + v_j)$$
证明唯一性：
将含有自由变量 $k$ 的通解代入 $M$ 的表达式：

$$M(k) = (u_0 + k\mathbf{1})\mathbf{1}^T + \mathbf{1}(v_0 - k\mathbf{1})^T$$

利用矩阵分配律展开：

$$M(k) = u_0\mathbf{1}^T + k(\mathbf{1}\mathbf{1}^T) + \mathbf{1}v_0^T - k(\mathbf{1}\mathbf{1}^T)$$

消去 $k$ 相关项：

$$M(k) = u_0\mathbf{1}^T + \mathbf{1}v_0^T = M_{\text{fixed}}$$
结论：
对于 $Ax=b$ 的任何解 $x$，由它们计算得到的矩阵 $M$ 是确定的。即：

$$M = f(R, b)$$

$M$ 是 $R$ 和 $b$ 的确定函数，不受具体解的影响；因此只要求解能收敛，就可以计算出正确的梯度。


## 收敛性

我们采取共轭梯度法求解这个线性系统。

### 1）特征值

若 $R$ 的奇异值为 $\sigma_1, \sigma_2, \dots, \sigma_n$，则 $A$ 的特征值为 $1 \pm \sigma_i$。

证明：设 $\mu$ 是 $A$ 的特征值，对应的特征向量为 $z = \begin{bmatrix} p \\ q \end{bmatrix}$，其中 $p, q \in \mathbb{R}^n$。

则有方程组：

- $p + Rq = \mu p \implies Rq = (\mu - 1)p$

- $R^Tp + q = \mu q \implies R^Tp = (\mu - 1)q$

将 (2) 代入 (1) 可得：

$$\dfrac{RR^Tp}{\mu - 1} = (\mu - 1)p \implies RR^T p = (\mu - 1)^2 p$$

这表明 $(\mu - 1)^2$ 是矩阵 $RR^T$ 的特征值。根据奇异值分解的定义，$RR^T$ 的特征值正是 $R$ 的奇异值的平方 $\sigma_i^2$。

因此：

$$(\mu - 1)^2 = \sigma_i^2 \implies \mu - 1 = \pm \sigma_i \implies \mu = 1 \pm \sigma_i$$

### 2）对称半正定性

$A$ 是对称半正定矩阵（Positive Semidefinite）。对称性是显然的。

$A$ 的特征值为 $1 \pm \sigma_i$，因为 $R$ 是bistochastic matrix，根据 Perron-Frobenius定理, 其最大奇异值 $\sigma_{\max}(R) = 1$。

由于所有奇异值 $\sigma_i$ 满足 $0 \le \sigma_i \le 1$，则：

- 最大特征值 $\mu_{\max} = 1 + \sigma_1 = 1 + 1 = 2$。

- 最小特征值 $\mu_{\min} = 1 - \sigma_1 = 1 - 1 = 0$。

因为所有特征值 $\mu_i \ge 0$，所以 $A$ 是半正定的。

### 3）相容性

共轭梯度法在处理奇异的对称半正定矩阵时，[只有满足下列条件才会收敛到解](https://arxiv.org/pdf/1809.00793)：

$$b \in \mathcal{R}(A) \iff b \perp \mathcal{N}(A)$$

其中 $\mathcal{N}(A)$ 是 $A$ 的零空间（Null Space）；这个条件被称为相容性条件。

针对该矩阵的相容性条件：

由于 $R$ 是bistochastic matrix，我们已知 $A \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = \mathbf{0}$。这意味着 $\begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix}$ 在零空间内。

设 $b = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$（$b_1, b_2 \in \mathbb{R}^n$），则根据相容性条件，

$$\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}^T \begin{bmatrix} \mathbf{1} \\ -\mathbf{1} \end{bmatrix} = 0 \implies \sum_{i=1}^n (b_1)_i = \sum_{j=1}^n (b_2)_j$$

而 $b_1 = (G \odot R) \mathbf{1}$，$b_2 = (G \odot R)^T \mathbf{1}$，因此

$$\sum_{i=1}^n (b_1)_i = \sum_{j=1}^n (b_2)_j = \sum_{i=1}^n \sum_{j=1}^n G_{ij} R_{ij}$$

满足相容性条件，算法应当收敛。

## torch实现

特别鸣谢Gemini的辅助编程。

cuTile的示例实现在[这里](https://gist.github.com/Da1sypetals/e9886cd679b32920100656d7a3dee79b).

> 注：要确保正向sinkhorn充分收敛才能使用此方法；正向收敛不充分的情况下，和自动求导的结果对比会出现很大偏差。

```py
from icecream import ic
import torch
import einops as ein

dtype = torch.float32

batch = 25001
n = 4
iters = 20
print(f"{n = }")
print(f"{iters = }")

# Fix torch seed
# torch.manual_seed(0)


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


def batch_cg_solve(R, b):
    """
    Solve the system Ax = b using the Conjugate Gradient (CG) method.
    The matrix A is structured as:
    A = [[I,   R ],
         [R^T, I ]]
    """
    batch_size, n, _ = R.shape
    device = R.device
    dtype = R.dtype

    # 1. Construct the complete 2n x 2n matrix A
    # Create identity matrix I
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # Concatenate blocks to form A
    # top: [I, R]
    top = torch.cat([eye, R], dim=-1)
    # bottom: [R^T, I]
    # Use einsum 'bij->bji' for transpose
    R_T = torch.einsum("bij->bji", R)
    bottom = torch.cat([R_T, eye], dim=-1)
    # A shape: (batch, 2n, 2n)
    A = torch.cat([top, bottom], dim=-2)

    # 2. CG Initialization
    # Initial guess x0 = 0, shape (batch, 2n)
    x = torch.zeros_like(b)

    # Initial residual r0 = b - A@x0 = b
    r = b.clone()

    # Initial search direction p0 = r0
    p = r.clone()

    # rs_old = r^T * r (dot product per batch)
    rs_old = torch.einsum("bi,bi->b", r, r)

    max_iter = 2 * n

    # 3. CG Iteration Loop
    for i in range(max_iter):
        # Calculate Ap = A @ p
        # 'bij,bj->bi' performs batch matrix-vector multiplication
        Ap = torch.einsum("bij,bj->bi", A, p)

        # Calculate step size alpha = (r^T * r) / (p^T * A * p)
        # pAp is the dot product of p and Ap per batch
        pAp = torch.einsum("bi,bi->b", p, Ap)
        # alpha = rs_old / pAp
        # Avoid division by zero here is very important
        alpha = rs_old / (pAp + 1e-12)

        # Update solution x = x + alpha * p
        # 'b,bi->bi' scales each vector in the batch by its corresponding alpha
        x += torch.einsum("b,bi->bi", alpha, p)

        # Update residual r = r - alpha * Ap
        r -= torch.einsum("b,bi->bi", alpha, Ap)

        # Calculate new residual inner product
        rs_new = torch.einsum("bi,bi->b", r, r)

        # Calculate beta = (r_new^T * r_new) / (r_old^T * r_old)
        # Avoid division by zero here is not so important experimentally
        # but it's good to have it
        beta = rs_new / (rs_old + 1e-12)

        # Update search direction p = r + beta * p
        p = r + torch.einsum("b,bi->bi", beta, p)

        rs_old = rs_new

    return x


def sinkhorn_backward_implicit(grad_R, R):
    R = R.detach()

    r = (R * grad_R).sum(dim=-1)  # shape (n,)
    c = (R * grad_R).sum(dim=-2)  # shape (n,)

    # Build 2n x 2n system
    A = torch.zeros((batch, 2 * n, 2 * n), dtype=dtype)

    A[:, :n, :n] = torch.eye(n, dtype=dtype).unsqueeze(0)
    A[:, :n, n:] = R
    A[:, n:, :n] = R.transpose(-2, -1)
    A[:, n:, n:] = torch.eye(n, dtype=dtype).unsqueeze(0)

    ic(torch.linalg.svdvals(A))

    b = torch.cat([r, c], dim=-1)

    ic(A.shape)
    ic(b.shape)

    # sol = torch.linalg.solve(A, b)
    sol = batch_cg_solve(R, b)

    alpha = sol[:, :n]
    beta = sol[:, n:]

    Gproj = grad_R - alpha.unsqueeze(-1) - beta.unsqueeze(-2)
    return Gproj * R


######################################################################
# Variable
######################################################################
dist = torch.distributions.uniform.Uniform(0.0, 4.0)
M = dist.sample((batch, n, n))
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

# KL pullback:
grad_M_implicit = sinkhorn_backward_implicit(grad_R, R)


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
max_abs_diff = abs_diff.reshape(batch, -1).max(-1).values.tolist()
mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
max_rel_diff = rel_diff.reshape(batch, -1).max(-1).values.tolist()

print(f"MAE: {format_list(MAE)}")
print(f"max_abs_diff: {format_list(max_abs_diff)}")
print(f"mean_rel_diff: {format_list(mean_rel_diff)}")
print(f"max_rel_diff: {format_list(max_rel_diff)}")

print(f"Max MAE = {max(MAE)}")
print(f"Max max_abs_diff = {max(max_abs_diff)}")
print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
print(f"Max max_rel_diff = {max(max_rel_diff)}")

print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
print("\nGrad (implicit) sample:\n", g2[0, :3, :3])
```