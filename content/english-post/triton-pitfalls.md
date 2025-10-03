+++
date = '2025-10-02T15:12:24+08:00'
title = 'Triton Common Pitfalls'
tags = ["deep-learning", "triton"]
+++

From the perspective of a newbie user

## The Documentation is a Disaster

Recently, I had to optimize a custom operator and decided to use OpenAI's Triton. After digging into the documentation, I was shocked at how poorly written it is — like an academic paper full of equations but lacking practical code examples.

If the library operates on tensors, the docs should clearly specify input/output shapes and provide concrete examples (like PyTorch does). Instead, everything is vaguely described in plain text, leaving users to guess the details.

## How Triton Fails at Clarity

Take the [tl.load documentation](https://triton-lang.org/main/python-api/generated/triton.language.load.html#triton.language.load) as an example. It mentions that block pointers support "boundary checks" and "padding options," but:

### What does "boundary check" actually do?
- Does it skip out-of-bounds elements, returning a smaller tensor?
- Does it pad with a default value?
- Does it throw an error?
- The docs don't say.

### What's the "padding option"?

After some trial and error, I realized it handles out-of-bounds elements — but this should be explicitly stated, not left for users to reverse-engineer.

Another issue: `tl.make_block_ptr` and `tl.arange` require block shapes and element counts to be powers of two. This restriction isn't mentioned anywhere in the official docs. I only discovered it after hitting an error and finding a passing reference in an unofficial blog post.

Whoever wrote this documentation did a huge disservice to the engineers who built Triton's compiler. Triton's compiler is awesome.

## Key API Clarifications

### tl.load

- For raw pointers (or tensors of pointers): Always set `mask` and `other`.
  - `mask=True`: Load from HBM.
  - `mask=False`: Use the value from `other` (a float).
- For block pointers (`tl.make_block_ptr`): Enable boundary checks on all dimensions and set `padding="zero"`. The behavior of `boundary_check` is poorly defined, especially after reordering dimensions.

## Shape Constraints

`tl.arange` element counts and `tl.make_block_ptr` block shapes must be powers of two. This might apply to all Triton tensor dimensions, but I haven't verified it.

## Memory Access Pitfalls

- `tl.load` and `tl.store` silently corrupt data. Invalid memory access turns values into `NaN`—yes, even `tl.store` can corrupt valid data!
- Solution: Unless your dimensions are multiples of 64, always enable boundary checks for HBM reads/writes.
- Extra caution: Raw pointers require careful `mask` handling to avoid disasters.