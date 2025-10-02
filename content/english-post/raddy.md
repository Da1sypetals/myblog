+++
date = '2025-10-02T15:15:21+08:00'
title = 'Raddy devlog: forward autodiff system'
tags = ["rust", "graphics", "math"]
+++

**TL;DR:** I created [Raddy](https://github.com/Da1sypetals/Raddy), a forward autodiff library, and [Symars](https://github.com/Da1sypetals/Symars), a symbolic codegen library.

If you're interested, please give them a star and try them out! ❤️

## The Origin of the Story

I recently read papers on physical simulation and wanted to reproduce them. I started with [Stable Neo-Hookean Flesh Simulation](https://graphics.pixar.com/library/StableElasticity/paper.pdf), though the choice isn't critical. Many modern physical simulations are implicit, requiring Newton's method to solve optimization problems.

This involves:
- Computing derivatives of the constitutive energy model (first-order gradient, second-order Hessian).
- Assembling a large, sparse Hessian from small, dense Hessian submatrices — a delicate task prone to hard-to-debug bugs.

From [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/), I learned deriving these formulas is labor-intensive (even understanding the notation takes time). Searching for alternatives to avoid meticulous debugging, I found two solutions:
- Symbolic differentiation with code generation.
- Automatic differentiation.

Tools for the former include MATLAB or SymPy; for the latter, deep learning libraries like PyTorch or more suitable ones like [TinyAD](https://github.com/patr-schm/TinyAD).

Why TinyAD? Deep learning libraries differentiate at the tensor level, but I needed scalar-level differentiation for physical simulations. Tensor-level differentiation could lead to unplayable frame rates.

A problem arose: these tools are in the C++ toolchain, and I'm not proficient in C++ (I know some kindergarten-level C++, but CMake and libraries like Eigen defeated me after three days of trying). So, I switched to Rust, a language I'm more comfortable with. This was the start of all troubles…

## A Path That Seems Simple

Rust lacks an automatic differentiation library for second-order Hessians (at least on crates.io). SymPy can generate Rust code, but it's buggy. Given the implementation complexity, I started with symbolic code generation, creating [Symars](https://github.com/Da1sypetals/Symars).

SymPy's symbolic expressions are tree-structured, with nodes as operators (`Add`, `Mul`, `Div`, `Sin`, etc.) or constants/symbols, and children as operands. Code generation involves depth-first traversal: compute child expressions, then the current node's expression based on its type. Base cases are constants or symbols.

I used the generated derivatives for a simple implicit spring-mass system, but debugging index errors in Hessian assembly was time-consuming.

## Trying the Untrodden Path Again

To address this, I revisited automatic differentiation, aiming to adapt TinyAD for Rust.

### Two Ways to Walk the Same Path

Initially, I considered two approaches:
- Write FFI bindings, as I don't know C++ well.
- Replicate TinyAD's logic.

Cloning TinyAD, I couldn't even pull dependencies or compile it. Examining the codebase, I found the core logic was ~1000 lines — manageable to replicate without running the project. Thus, [Raddy](https://github.com/Da1sypetals/Raddy) was born.

## Symbolic diff & Codegen: Implementation

Implementation details:
- Each scalar in the differentiation chain carries a gradient and Hessian, increasing memory overhead. I avoided implementing the `Copy` trait, requiring explicit cloning.
- Operator traits between `(&)Type` and `(&)Type` (four combinations) required repetitive code. I considered the following options:
  - Macros.
  - Python scripts for code generation.

Macros breaks `rust-analyzer` (somebody refuse to agree on this, but for me this is true) and I am rather unfamiliar with Rust's macro syntax, so I used Python scripts (in the `meta/` directory) for simple string concatenation.

- Testing: I verified derivatives by generating symbolic `grad` and `hessian` code with Symars, cross-validating against Raddy's results, ensuring test expressions covered all implemented methods. Symars performed reliably, without bugs.

## What about sparse matrices

Dense matrices store adjacent values contiguously, but sparse matrices (with millions of elements) don't. I implemented sparse Hessian assembly:

- Define a problem via the `Objective<N>` trait:
- Specify problem size `N` (a compile-time constant for const generics).
- Implement computation logic, e.g., a spring-mass system (Hooke's law, E=1/2 k x²):

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

- Specify input components' indices (`&[[usize; N]]`).
- Automatically assemble sparse `grad` and `hess` (handling index mapping).
- Manually sum multiple `grad` and `hess` (simple matrix addition; triplet matrices are concatenated).

Before tests, Raddy was 2.2k lines; after, it ballooned to 18k lines, showing LOC is a poor metric.

Finally, I wrote a demo for fun and as an example.

![](../images/raddy-mass-spring.gif)

## Conclusion

Gains:
- Learned how automatic differentiation works.
- First time using AI for documentation (it struggled with Rust syntax, producing test code with errors).
- Happiness!