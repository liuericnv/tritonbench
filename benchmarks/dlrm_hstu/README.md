# Randomized HSTU branch validation

Install the matching Meta Triton development branch and initialize TritonBench's
`generative-recommenders` submodule. Use one visible B200 or B300 and a new output
directory for every invocation:

```bash
python benchmarks/dlrm_hstu/validate.py \
  --meta-triton /path/to/meta-triton --base-commit 7a8758d7c92a24b0bccdadfd350fb7d65994cfca \
  --seq 4096 --seeds 10 --batch 512 --fp32-sequences 8 \
  --output /path/to/new-output/seq-4096
```

Repeat for maximum lengths 256, 512, 1024 and 2048. `--case-index 19` isolates the
previous long-sequence dQ regression input (seed 4, uniform [-2, 2]); sequence 304 is
always included among the FP32 checks at 4096. The harness also includes boundary
and random sequences. The requested eight random FP32 sequences are additional
to mandatory boundary/regression sequences, with duplicates removed.

Every invocation generates 40 randomized full-B512 fixtures plus one zero
control. Inputs cover Gaussian scales 0.01, 1 and 4, uniform [-2, 2], packed and
contiguous QKV, varying ragged lengths and target counts, and independent random
upstream gradients. The unchanged upstream kernel is loaded from the explicit
Git base commit in a separate module. Both actual TritonBench providers run
forward and backward on the same full-batch tensors.

Full-batch comparison includes all output and gradient elements in every
sequence, using the historical 2% sequence / 3% final-row L2 limits. The hybrid must
also satisfy **1% per-sequence and 1% final-row L2 against independent FP32** on all
selected sequences, for output, dQ, dK and dV. Upstream and ordinary Hammer FP32
errors are diagnostic, so existing baseline rounding is visible rather than
treated as exact truth. FP32 math implements SiLU and the causal-history plus
independent-target mask without importing kernel math helpers.

`checks.jsonl` retains every numerical result and selected sequence index.
`summary.json` records completion and failure counts; `provenance.json` records
source commits, installed kernel hash, runtime versions and GPU. This program
does not collect performance timings. Normal `run.py` benchmark commands are
documented in `tritonbench/operators/ragged_attention/README.md`.
