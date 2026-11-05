# HSTU providers

`hstu_triton_hammer` runs the ordinary Meta Triton HSTU tutorial configuration.
`hstu_triton_hybrid` uses its autotuned forward and the repaired TLX backward
with FP32 dQ scratch on supported Blackwell BF16 inputs. Other head dimensions,
value dimensions, window/context masks and scale layouts retain the tutorial's
general backward path. The hybrid requires the updated Meta Triton tutorial
with `HSTU_SILU_BWD_API_VERSION = 1`.

The hybrid loads a separate instance of the same tutorial so it has independent
configuration and autotuners. Constructing or running it does not switch the
Hammer provider to the optimized backward. No tile is pinned to the production
benchmark shape. Short sequences can favor the ordinary provider.

Build/install the matching Meta Triton `dlrm-dev` branch, initialize the
`generative-recommenders` submodule, and run from this repository. For example:

```bash
python run.py --op ragged_attention --precision bf16 \
  --batch-size 512 --heads 4 --attn-dim 128 --hidden-dim 128 \
  --min-seq-len-log2 12 --max-seq-len-log2 12 \
  --seq-sparsity 0.95 --target-size 20 \
  --only hstu_triton_hammer,hstu_triton_hybrid --mode fwd \
  --warmup 10000 --rep 20000 --metrics latency
```

Use `--mode bwd` for whole-backward timing. Change both sequence logarithms to
8, 9, 10 or 11 for maximum lengths 256, 512, 1024 or 2048. `--warmup` and `--rep` are
millisecond budgets, not fixed iteration counts. Graph replay is disabled by
default. For accepted results, use three fresh processes and save each output
separately. Correctness coverage lives in
`test/test_gpu/test_hstu_hybrid.py` and the Meta Triton HSTU regression suite.
