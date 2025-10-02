+++
date = '2025-10-02T15:01:14+08:00'
title = 'SnapViewer Devlog #3: Optimizations'
tags = ["torch", "deep-learning", "rust"]
+++

**Intro: Troubleshooting Memory and Speed Performance**

**Disclaimer:** I develop and test primarily on Windows using the latest stable Rust toolchain and CPython 3.13.

## 1. Background and Motivation

SnapViewer handles large memory snapshots effectively — for example, pickle files up to 1 GB and compressed snapshots up to 500 MB. However, when processing extremely large dumps (e.g., a 1.3 GB snapshot), we encountered serious memory and speed bottlenecks:

- Format conversion (pickle → compressed JSON) triggered memory peaks around 30 GB.
- Data loading of the compressed JSON into Rust structures caused another ~30 GB spike.

Frequent page faults and intense disk I/O (observed in Task Manager) made the application sluggish and prone to stalls. To address this, we applied a Profile-Guided Optimization (PGO) approach.

## 2. Profile-Guided Optimization

PGO requires empirical profiling to identify the true hotspots. I began with memory profiling using the [memory-stats](https://crates.io/crates/memory-stats) crate for lightweight inspection during early optimization stages. Then, I decomposed the data-loading pipeline into discrete steps:

- Reading the compressed file (heavy disk I/O)
- Extracting the JSON string from the compressed stream
- Deserializing the JSON into native Rust data structures
- Populating an in-memory SQLite database for ad-hoc SQL queries
- Building the triangle mesh on CPU
- Initializing the rendering window (CPU-GPU transfer)

Profiling revealed two major memory culprits: excessive cloning and multiple intermediate data structures. Below, I outline the optimizations.

### Eliminating Redundant Clones

During rapid prototyping, calls to `.clone()` are convenient. But they are expensive. Profiling showed that cloning large vectors contributed significantly to the memory peak and CPU time.

- **First attempt:** switch from cloned `Vec<T>` to borrowed `&[T]` slices. This failed due to lifetime constraints.
- **Final solution:** use `Arc<[T]>`. Although I'm not leveraging multithreading, `Arc` satisfies PyO3's requirements, while no significant overhead is observed in this context.

This change alone reduced memory usage and improved throughput noticeably.

### Early Deallocation of Intermediate Structures

Constructing the triangle mesh involved several temporary representations:

- Raw allocation buffers
- A list of triangles (vertices + face indices)
- A CPU-side mesh structure
- GPU upload buffers

Each stage held onto its predecessor until the end of scope, inflating peak usage. To free these intermediates promptly, the following is implemented:

- Scoped blocks to limit lifetimes
- Explicitly invoked `drop()` on unneeded buffers

After these adjustments, peak memory dropped by roughly one-third.

## 3. Sharding JSON Deserialization

Deserializing the call-stack JSON with over 50 000 entries spiked memory usage dramatically. To mitigate this:

- Shard the JSON data into chunks of at most 50 000 entries.
- Deserialize each chunk independently.
- Concatenate the resulting vectors.

This streaming approach kept per-shard memory small, eliminating the previous giant allocation.

> It is worth noting that `serde_json::StreamDeserializer` can be another option worth trying.

## 4. Redesigning the Snapshot Format

Even after the above optimizations, the call-stack data remained the largest in-memory component — duplicated once in Rust and again in the in-memory SQLite database.

To remove redundancy, I rethought what each representation serves:

- **Rust structures:** display call stacks on screen upon user click.
- **SQLite DB:** serve ad-hoc SQL queries.

Since SnapViewer is single-threaded and can tolerate occasional disk I/O, I split the snapshot into two files:

- **allocations.json:** lightweight JSON with allocation timestamps and sizes.
- **elements.db:** SQLite database holding call-stack text (indexed by allocation index).

These two files are zipped together. At runtime:

- Unzip the snapshot.
- Load `allocations.json` into memory (small footprint).
- Open `elements.db` on disk.
- On click, query `elements.db` with `WHERE idx = <allocation_index>`.

SQLite's efficient on-disk indices make these lookups fast, with no perceptible impact on frame rate.

### Refactoring the Conversion Script

I updated the snapshot-conversion script as follows:

- Parse the original snapshot format.
- Bulk-insert call stacks into an in-memory SQLite database, then dump the DB as a byte stream.
- Serialize allocation metadata to JSON.
- Zip the JSON and DB byte stream.

While conversion takes slightly longer, the resulting snapshot loads faster and uses a fraction of the memory.

## 5. Results and Lessons

After these optimizations, SnapViewer:

- No longer spikes to 60+ GB of RAM on large snapshots, since we do not load the entire call stack information into memory at all.
- Starts up much faster.
- Maintains smooth rendering, even with on-demand call-stack queries.

What I learned:

- Do not always load everything into memory. When you overflow your memory, the performance of virtual memory swapping system is probably worse than you think.
- When you need some mechanism to store most data on disk, but intelligentlly cache some of then in memory, SQLite should be a good start. It has its well-designed and industry-proven algorithm built into it.