+++
date = '2024-12-29'
title = '自动求导, 道阻且长'
+++

TL;DR: [Symars ](https://github.com/Da1sypetals/Symars) Rust代码生成库和 [Raddy](https://github.com/Da1sypetals/Raddy) 自动求导库的来龙去脉

## 故事的起因：

前段时间读了一些物理模拟的论文，想尝试复现一下。下手点先选了 [stable neo hookean flesh simulation](https://graphics.pixar.com/library/StableElasticity/paper.pdf)，但是选了什么并不重要。重要的是，所谓“现代”的物理模拟很多是隐式模拟，需要用牛顿法解一个优化问题。

这之中就涉及到了：对能量的本构模型求导数（一阶梯度，二阶 hessian 矩阵）。这之中还涉及到从 _小而稠密_  的 hessian 子矩阵组装成 _大而稀疏_ 的完整 hessian。这是一个精细活，一不小心就会出现极其难以排查的 bug。

从 [*Dynamic Deformables*](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/) 这篇文章中可以看出推导这个公式就要花不少功夫（就算是看懂论文里的 notation 也要好一会儿），于是我搜了搜更多东西，尝试寻找一些其他的解决方法：我不是很想在精细的 debug 上花很多时间。最终找到的解决方法有两种：
- 求符号导数，然后进行代码生成；
- 自动求导。

找到的资料中，前者有 MATLAB 或者 SymPy，后者有 PyTorch 等深度学习库，和更适合的 [TinyAD](https://github.com/patr-schm/TinyAD)。
> 为什么说更适合？因为深度学习库的求导是以tensor为单位的，但是我这里的求导需要以单个标量为单位，粒度不同，深度学习库可能会跑出完全没法看的帧率。

但是一个致命的问题来了：上述工具都在 C++ 的工具链上，而我不会 C++（或者，我可能会一点点 C++，但是我不会 CMake，因此不会调包。）
>我曾经花了三天尝试在项目里用上 Eigen，然后失败告终，还是技术水平太菜了。

我只好换一门我比较熟悉的语言：Rust。这是一切罪恶的开始...

## 一条看起来简单的路

目前 Rust 还没有一个可以求二阶 hessian 的自动求导库（至少我在 crates.io 没搜到）。  
SymPy 目前还不能生成 Rust 代码（可以，但是有 bug）。  
考虑实现难度我先选了后者：从 SymPy 表达式生成 Rust 代码。于是有了 [Symars](https://github.com/Da1sypetals/Symars)。

SymPy 提供的访问符号表达式的数据结构是树的形式，节点类型是运算符类型（`Add`, `Mul`, `Div`, `Sin`, 等等）或者常数/符号，节点的孩子是 operand 操作数。实现代码生成的思路就是按深度优先遍历树，得到孩子的表达式，然后再根据节点类型得到当前节点的表达式。边界条件是当前节点是常数，或者符号。

实现完了之后，我拿着生成的导数去先写一个简单的隐式弹簧质点系统；但是还是在 hessian 组装上消耗了很多时间在排查 index 打错这种 bug 上。

## 再去走没走过的路

为了解决上述问题，我打算尝试原来放弃的那条路：自动求导。方案是在 Rust 里面使用 TinyAD。

### 一条路的两种走法

一开始想了两个方法：毕竟我不是很懂 C++，可能相比于看懂整个 TinyAD 的 codebase，做一套 FFI 更现实一些。

但是我发现，项目 clone 下来之后，我甚至不会拉依赖不会编译。（什么赛博残废）

然后我重新观察了 TinyAD 的 codebase，发现核心逻辑大概在 ~1000 行代码，似乎不是不可能在完全不运行这个项目的前提下把代码复刻一遍。说干就干，于是有了[Raddy](https://github.com/Da1sypetals/raddy)：

### 正确的走路姿势

找到了正确的走路姿势，开始着手实现。说一些实现细节：
- 每个求导链路上的标量值都带一个相对变量的梯度和 hessian，所以肉眼可见的 memory overhead 比较严重；一个提醒用户的方法是不实现 `Copy` trait，在需要一个副本的时候 `explicit clone`。
- 有大量需要实现 `(&)Type` 和 `(&)Type` 之间的 operator trait，组合有 `2 * 2 = 4` 种，这意味着相同的代码要写 4 次。于是考虑引进某些元编程的方法：
  - 用宏 `macro` 批量实现；
  - 用 Python 脚本进行代码生成。

考虑到宏会让 `rust-analyzer` （局部）罢工，但是我离开 LSP 简直活不了，于是选择了后者。具体代码见 `meta/` 目录，其实没啥技术含量，就是字符串拼接。

- 测试：我要如何验证我求出来的导数是对的？第一个想法就是用我前面写过的 `symars`，对每个测试表达式生成其符号 `grad` 和 `hessian` 的代码，然后和求导结果交叉验证，然后让这些测试表达式尽可能覆盖所有实现过的方法。
  - `symars` 居然表现得很不错，稳定使用没有发现 bug。

## 稀疏之路

稠密的矩阵用一块连续的内存空间表示相邻的值；稀疏矩阵动辄上万的边长（上亿的总元素数 `numel`）不允许。于是针对稀疏矩阵单独实现了其 hessian 的组装过程：

- 定义一个问题，即实现一个 `Objective<N>` trait，需要：
  - 确定 problem size `N`（这是编译器要求 const generics 必须是编译期常量）
  - 实现计算逻辑
  - 比如：弹簧质点系统的逻辑（其实就是高中学的胡克定律 $E =\dfrac{1}{2}kx^2$ ）：
    - 简单解释：在二维平面中模拟，每个点坐标 $(x,y)$ 有两个实数；每个弹簧涉及两个点，得到 $2 \times  2 =4$ 这个自由度。
    ```rust
    impl Objective<4> for SpringEnergy {
        type EvalArgs = f64; // restlength

        fn eval(&self, variables: &advec<4, 4>, restlen: &Self::EvalArgs) -> Ad<4> {
            // extract node positions from problem input:
            let p1 = advec::<4, 2>::new(variables[0].clone(), variables[1].clone());
            let p2 = advec::<4, 2>::new(variables[2].clone(), variables[3].clone());

            let len = (p2 - p1).norm();
            let e = make::val(0.5 * self.k) * (len - make::val(*restlen)).powi(2);

            e
        }
    }
    ```

- 定义这个稀疏向量中的哪些分量，需要作为这个问题的输入（提供其 indices，`&[[usize; N]]`）。
- AD 自动组装 `grad` 和 `hess`（稀疏），涉及到 index map 的问题；
- 最后用户手动将多个 `grad` 和 `hess` 加和。这一步就没有 index map 的问题了，就是简单的矩阵加法（triplet matrix 就更简单，直接把多个 triplet vector 接在一起就好了）。

添加测试之前总共有2.2k行代码，添加测试之后项目总代码量膨胀到了18k行，再次证明数LOC是个没啥用的事情。

最后，经过一大堆冗长的测试，写了一个 demo 来娱乐自己，顺便作为 example：
![spring](../images/spring.gif)


# 结语

收获：
- 熟悉了自动求导
- 用 AI 写文档（他目前还读不懂我的代码，或者说还读不太懂 Rust，所以写的测试有许多语法问题）
- Happiness!