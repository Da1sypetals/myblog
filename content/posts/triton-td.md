+++
date = '2025-10-31T21:49:36+08:00'
title = 'Triton Tensor Descriptor: 茴字的第三种写法'
+++

今天我们来介绍~~茴字的第3种写法~~ 

今天我们来介绍 Triton 中的第三种进行 tensor 指针运算的 API：Tensor Descriptor。内容来自[triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)。

## 关于 triton 的基本概念

- triton 只是和 python 共用语言前端（我们写的代码），triton 会接管 python 的 AST，然后后续步骤就交由 triton 编译器一步步 lower 到 GPU 代码了。
- 在第一次执行一个 kernel 之前发生的事情称为编译期，之后的执行称为运行时。
- triton的 kernel launch 的grid 参数是一个 ndrange，在 kernel 里面获取到的 `program_id(i)` 就是第 i 维度的 index。



## Tensor Descriptor的用法

### 创建

```python
desc = tl.make_tensor_descriptor(
    pointer,
    shape=[M, N],
    strides=[N, 1],
    block_shape=[M_BLOCK, N_BLOCK],
)
```

其中：
- `pointer` 就是传入triton kernel的tensor
- `shape` 是一个整数列表，**可以编译期确定，也可以运行时动态传入，可以不是tilesize的倍数**
    - 传入 `[tensor.shape(i) for i in range(tensor.dim())]`
- `strides` 是一个整数列表，**可以编译期确定，也可以运行时动态传入，可以不是tilesize的倍数**
    - 传入 `[tensor.stride(i) for i in range(tensor.dim())]`
- `block_shape` 是一个整数列表，**必须是编译期常量**
    - 对应概念是CUDA的blockDim
- 上述三者的长度必须相同，等于输入tensor的 `.dim()`

### 读写

#### 读
```python
value = desc.load([moffset, noffset])
```

其中唯一的参数offsets是一个整数列表：
- **可以编译期确定，也可以运行时动态传入**
- 列表里面每个值是对应维度的**元素级别** offset

#### 写
```python
desc.store([moffset, noffset], tl.abs(value))
```

其中：
- 第一个参数offsets是一个整数列表，和读一样：
    - **可以编译期确定，也可以运行时动态传入**
    - 列表里面每个值是对应维度的**元素级别** offset
- 第二个参数是一个 buffer，shape 必须和 `make_tensor_descriptor` 的时候指定的 `block_shape` 相同

## 例子

请一行代码一行代码读过去，你一定能看懂的。

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))


M, N = 256, 256
x = torch.randn(M, N, device="cuda")
M_BLOCK, N_BLOCK = 32, 32
grid = (M / M_BLOCK, N / N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```