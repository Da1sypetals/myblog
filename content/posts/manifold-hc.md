+++
date = '2026-01-04T19:55:33+08:00'
title = 'DeepSeek mHC的简单演示（可能有错误）'
+++

DeepSeek发布了最新的魔改版Residual Connection：Manifold Constrained Hyper-Connection.

其基本思路是把旁路residual限制在某个集合上（文中用更“几何”的manifold一词显示，退化的例子就是Kaiming的原版Residual Connection，约束是`residual = x`），使其在正反向传播的时候具有某种不易爆炸/崩溃的数学性质。

类似的思路还可以在比如物理模拟中看到：
- 通过将物体的transformation matrix约束在$SE(3)$，禁止物体形变，从而模拟刚体。
- 进一步，在Affine body dynamics中，通过一个惩罚项惩罚transformation matrix偏离$SE(3)$的部分，将物体的transformation映射到尽可能近的$SE(3)$，从而在物体的行为尽可能接近刚体的同时，解决系统难以求解（有约束 $\rightarrow$无约束）的问题。

HC的基本思路应该是：

- 原本就有n个stream
- 在主线forward的时候，把n个stream合并为一个（pre-proj），通过这一层网络（$f$），然后再打散回n个stream（post-proj）
    - 即 $y=\text{post-proj} \circ f \circ \text{pre-proj}(x)$ 
- 支线复制输入x，通过一个res-proj进行信息混合之后，加回主线的输出

mHC对这个res-proj进行约束：
- 要求其为doubly stochastic matrix. 
- 具体做法就是通过sinkhorn迭代直接将其映射到最接近的doubly stochastic matrix上。

我自己写的，可能有错误的简单的代码实现[在这里](https://gist.github.com/Da1sypetals/0a7f70bf6b4ca7d46f0a1c5910e1a8b6)：

```py
# Reference: https://www.arxiv.org/abs/2512.24880
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ein
from icecream import ic

N_ITER = 20


def sinkhorn_knopp(mat: torch.Tensor) -> torch.Tensor:
    """
    mat: (..., n, n)
    Sidenote: IMO this technique should be subject to frequent change if mHC is proved to be effective
    """
    for _ in range(N_ITER):
        mat = mat / mat.sum(-2, keepdim=True)  # column normalize
        mat = mat / mat.sum(-1, keepdim=True)  # row normalize

    return mat


n = 4  # stream width
C = 256  # embedding dim

norm = nn.RMSNorm((n * C,))

phi_pre = nn.Parameter(torch.randn(n * C, n))
phi_post = nn.Parameter(torch.randn(n * C, n))
phi_res = nn.Parameter(torch.randn(n * C, n * n))

b_pre = nn.Parameter(torch.randn(1, n))
b_post = nn.Parameter(torch.randn(1, n))
b_res = nn.Parameter(torch.randn(n, n))


alpha_pre = nn.Parameter(torch.tensor(0.1))  # for dimension illustration purposes
alpha_post = nn.Parameter(torch.tensor(0.1))  # for dimension illustration purposes
alpha_res = nn.Parameter(torch.tensor(1.0))  # for dimension illustration purposes


def broadcast_to_n_stream(xl: torch.Tensor) -> torch.Tensor:
    return ein.repeat(xl, "... C -> ... n C", n=n)


def reduce_to_one_stream(xl: torch.Tensor) -> torch.Tensor:
    return ein.reduce(xl, "... n C -> ... C", "mean")


def manifold_constrained_hyperconnection(xl: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    # x: (..., n, C)

    # ===== residual =====
    xl_vec = ein.rearrange(xl, "... n C -> ... (n C)")
    xl_vec_prime = norm(xl_vec)  # (..., n*C)

    # data dependent mapping construction
    h_tilde_pre = alpha_pre * (xl_vec_prime @ phi_pre) + b_pre  # (..., n)
    h_tilde_post = alpha_post * (xl_vec_prime @ phi_post) + b_post  # (..., n)
    h_tilde_res = (
        alpha_res
        * ein.rearrange(
            (xl_vec_prime @ phi_res),
            "... (m n) -> ... m n",
            n=n,
        )
        + b_res
    )  # (..., n, n)

    h_pre = F.sigmoid(h_tilde_pre)  # (..., n)
    h_post = 2 * F.sigmoid(h_tilde_post)  # (..., n)
    h_res = sinkhorn_knopp(h_tilde_res.exp())  # (..., n, n)

    ic(h_pre.shape)
    ic(h_post.shape)
    ic(h_res.shape)

    # data dependent mapping application
    residual = ein.einsum(h_res, xl, "... m n, ... n C -> ... m C")  # m=n
    ic(residual.shape)

    # ===== mainstream =====

    x_pre = ein.einsum(h_pre, xl, "... n, ... n C -> ... C")
    ic(x_pre.shape)
    layer_out = layer(x_pre)  # (..., C)
    ic(layer_out.shape)
    x_post = ein.einsum(h_post, layer_out, "... n, ... C -> ... n C")
    ic(x_post.shape)

    out = x_post + residual  # (..., n, C)

    return out


batch_dims = (2, 100)
layers = [nn.Identity() for _ in range(3)]  # for illustration purpose

if __name__ == "__main__":
    x = torch.randn(*batch_dims, C)
    xl = broadcast_to_n_stream(x)
    # simulate 3 layers
    print("===== layer 1 =====")
    xl = manifold_constrained_hyperconnection(xl, layers[0])
    print("===== layer 2 =====")
    xl = manifold_constrained_hyperconnection(xl, layers[1])
    print("===== layer 3 =====")
    xl = manifold_constrained_hyperconnection(xl, layers[2])

    out = reduce_to_one_stream(xl)
    print("===== output =====")
    ic(out.shape)
```