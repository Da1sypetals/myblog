+++
date = '2025-10-01T16:09:53+08:00'
title = 'SnapViewer: Faster PyTorch Memory Allocation Viewer'
tags = ["torch", "deep-learning", "rust"]
+++

# Background

When training models with PyTorch, out-of-memory (OOM) errors are common, necessitating GPU memory optimization. When simple methods (like reducing batch size) no longer work, analyzing the memory footprint of the model itself may be required.

At this point, you might come across this [documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html), which teaches you how to record a memory snapshot and visualize it on this website.

However, there’s a major issue: the website is extremely laggy. If your model is small, with snapshots of just a few MB, the performance is somewhat tolerable. But if your model is large, with snapshots reaching tens or even hundreds of MB, the website becomes unbearably slow, with frame rates dropping as low as 2–3 frames per minute (this is not a typo).

I looked into the website’s JavaScript code, and here’s what it primarily does:

1. Manually loads Python pickle files;
2. Re-parses the raw data into graphical representations time the viewport changes, then renders it to the screen.

This parsing logic is written in JavaScript. You can imagine the performance when it is executed each frame, operating on hundred-MB data.

# Inspiration

My current work includes optimizing a deep learning model whose optimization is under-explored compared to LLM. I encountered this issue while working with a snapshot of a model with several billion parameters.

Why not just use existing LLM infrastructure instead of optimizing manually? Long story short, this model was custom-designed by a researcher and contains many modules completely different from standard LLMs. It seems like nowadays, everyone assumes deep learning is all about LLMs — so much so that even some tech leads believe LLM infrastructure can be easily adapted to other models… but I digress.
I originally wrote a simple script to parse the snapshot’s contents, hoping to identify memory allocation issues in the model. But after working with this model for a month, I finally had enough. That’s how this project — SnapViewer — came to be.

TL;DR​​: The graphical data from the memory snapshot is parsed and represented as a massive triangle mesh, leveraging existing rendering libraries to handle mesh rendering efficiently.

Here’s a snapshot of over 100 MB running smoothly on my integrated GPU:

![snapviewer](../images/snapviewer.gif)


# Implementation

## The reference implementation

The snapshot format is _partially_ documented in the record_memory_history function's [docstring](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py). However, this documentation is incomplete — likely because later updates weren’t reflected in the docstring.

The actual parsing of the snapshot into a dictionary happens [here](https://github.com/pytorch/pytorch/blob/main/torch/cuda/_memory_viz.py).
1. This script converts the allocator trace into a memory timeline, which is then passed to the web viewer’s JS code. 
2. The JS code further transforms this into polygons (representing allocations) for visualization. Each polygon corresponds to an allocation, storing details like size and callstack.

## Implementation: Snapshot (De)serialize

### Initial implementation
This part is impelmented in Python since I need to deal with Python-native data structures. I simply convert the dict to a json file.

### Optimizations
1. Raw JSON is too large on disk → compress it in-memory (Python zipfile) before writing.
2. During visualization, read the ZIP from disk (Rust zip crate) and decompress in-memory.

#### Tradeoffs
- This approach causes a temporary memory spike during JSON parsing but avoids persistent high memory usage.
- Also leverages Rust’s serde-json (since Rust’s serde-pickle is incomplete and can’t handle recursive structures).

## Implementation: Rendering & Interaction​​

This part is implemented in Rust.

### Rendering
- Since allocation data remains static during visualization, all allocations are combined into a single large mesh and sent to the GPU once.

- ​Library Used​​: three-d
    - Provides good mesh abstraction.
    - Supports one-time GPU upload (no per-frame CPU→GPU transfers).
    - Handles mouse/keyboard events.

### ​World-to-Window Coordinate Conversion​​
1. ​Step 1​​: Convert window coordinates to world coordinates (scale + window center offset).
2. ​​Step 2​​: Convert world coordinates to memory positions (predefined scaling).

### UI & Interaction Features​

#### Memory Scale Markers​​
- Dynamically adjust the number and precision of markers based on screen visibility.
- Keep markers at consistent screen positions while moving/zooming.

#### Pan & Zoom​​

1. Track the original scale (1/zoom).
2. Update to the new zoom level and compute the ratio between old and new scales.
3. Adjust the screen center position based on the mouse’s invariant world position.

## Implementation: Query

After using this tool at work for around a week, I find myself frequently needing to search in the memory snapshot, especially:

- Find all allocations which is alive at a specific timestamp
- Find all allocations whose call stack has a specific substring
- Preferablly the allocations should be sorted by allocation size in descending order


My first thought was to build a simple REPL and a simple command parser, and map each command to a specific query function.

However, after having listed out all the functionalities I want, I found it to be a subset of database query, especially SQL.

So I decided not to reinvent wheels: I just connect to a in-memory SQLite database. Interfacing user is simple: read user input, let SQLite execute it and format the output to human-readable format.

---

If you’ve struggled with PyTorch memory snapshots, [check it out](https://github.com/Da1sypetals/SnapViewer)! Contributions & feedback welcome. ⭐