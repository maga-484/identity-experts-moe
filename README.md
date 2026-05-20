# Identity Experts MoE

Sparse Mixture of Experts (MoE) with O(1) hierarchical hash routing. Replaces trainable gating networks with deterministic lookup tables to reduce compute overhead and VRAM usage.

## Problem

Dense neural networks activate 100% of parameters per inference. For hierarchical identity structures (`a.b.c.d.e.f.g.h`), this causes:

- Gradient conflicts across unrelated domains
- Memory bottlenecks at scale
- Linear latency growth with parameter count

## Solution

- **One tiny expert per identity node**
- **O(1) hash-based routing** (no matrix multiplication at gate)
- **Sparse activation**: only the target sub-network loads into VRAM

## Tech Stack

- Python 3.10+
- PyTorch
- NumPy, Pandas

## Installation

```bash
git clone https://github.com/maga-484/identity-experts-moe.git
cd identity-experts-moe
pip install -r requirements.txt
```

## Project Structure

```
docs/        # Technical documentation
models/      # PyTorch model definitions
results/     # Training metrics and benchmarks
src/         # Core code (dataset, router, training)
```

## Results

See results/ for loss curves and inference benchmarks.

## License

MIT License

Copyright (c) [2026] [Magali O.Gafe]
