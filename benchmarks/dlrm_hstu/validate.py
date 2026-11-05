"""Randomized whole-batch HSTU comparisons with upstream and streamed FP32.

Run with one CUDA GPU and the matching Meta Triton branch installed. Full-batch
comparisons include every sequence. FP32 checks stream selected independent
sequences to keep long-sequence checks practical, with the known dQ tail included.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-triton", type=Path, required=True)
    parser.add_argument("--base-commit", default="origin/main")
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--fp32-sequences", type=int, default=8)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.seq >= 21 and args.batch > 0 and args.seeds > 0
    assert args.fp32_sequences > 0
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import triton
    from tritonbench.operators.ragged_attention import operator
    from tritonbench.utils.parser import get_parser

    assert (
        torch.cuda.device_count() == 1 and torch.cuda.get_device_capability()[0] == 10
    )
    assert operator.HAS_HSTU_SELF_HYBRID, "Install the matching Meta Triton branch"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = git(args.meta_triton, "rev-parse", args.base_commit)
    relative = "third_party/tlx/tutorials/hstu_self_attn"
    # Load unchanged upstream independently. Pure helper sources are unchanged
    # by this branch and must agree before sharing their imports.
    for helper in ["stubs.py", "triton_attention_utils.py"]:
        upstream = subprocess.check_output(
            ["git", "-C", str(args.meta_triton), "show", f"{base}:{relative}/{helper}"]
        )
        assert upstream == (args.meta_triton / relative / helper).read_bytes()
    installed = Path(operator.hstu_self_triton_mha.__code__.co_filename)
    assert (
        hashlib.sha256(installed.read_bytes()).digest()
        == hashlib.sha256(
            (args.meta_triton / relative / "triton_hstu_attention.py").read_bytes()
        ).digest()
    )
    provenance = dict(
        meta_commit=git(args.meta_triton, "rev-parse", "HEAD"),
        base_commit=base,
        benchmark_commit=git(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        torch=torch.__version__,
        triton=triton.__version__,
        compiler_path=triton.__file__,
        installed_kernel=str(installed),
        installed_kernel_sha256=hashlib.sha256(installed.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),
        batch=args.batch,
        nmax=args.seq,
        seeds=args.seeds,
        fp32_sequences_requested=args.fp32_sequences,
        graph_replay=False,
        timings_collected=False,
        gates=dict(
            direct_sequence_l2=0.02, direct_last_row_l2=0.03, hybrid_fp32_l2=0.01
        ),
    )
    save(args.output / "provenance.json", provenance)

    def metric(got, expected):
        a, b = got.detach().float(), expected.detach().float()
        d = a - b
        return dict(
            finite=bool(a.isfinite().all() & b.isfinite().all()),
            relative_l2=float(d.norm() / b.norm().clamp_min(1e-30)),
            last_row_relative_l2=float(d[-1].norm() / b[-1].norm().clamp_min(1e-30)),
            max_absolute_error=float(d.abs().max()),
        )

    def full_metric(got, expected, ids, ends):
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype == torch.bfloat16
        a, b = got.detach().float(), expected.detach().float()
        row_error = (a - b).square().sum((1, 2))
        row_norm = b.square().sum((1, 2))
        error = torch.zeros(args.batch, device="cuda").scatter_add_(0, ids, row_error)
        norm = torch.zeros_like(error).scatter_add_(0, ids, row_norm)
        sequence_l2 = (error / norm.clamp_min(1e-30)).sqrt()
        last_l2 = (row_error[ends - 1] / row_norm[ends - 1].clamp_min(1e-30)).sqrt()
        return dict(
            finite=bool(a.isfinite().all() & b.isfinite().all()),
            max_sequence_relative_l2=float(sequence_l2.max()),
            max_last_row_relative_l2=float(last_l2.max()),
            relative_l2=float((error.sum() / norm.sum().clamp_min(1e-30)).sqrt()),
        )

    profiles = [
        ("normal", 0.01, "packed"),
        ("normal", 1.0, "contiguous"),
        ("normal", 4.0, "packed"),
        ("uniform", 2.0, "packed"),
    ]
    cases = [(seed, *profile) for seed in range(args.seeds) for profile in profiles] + [
        (0, "zero", 0.0, "packed")
    ]
    indexes = list(range(len(cases))) if args.case_index is None else [args.case_index]
    assert all(0 <= index < len(cases) for index in indexes)
    records = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="dlrm-upstream-") as temporary:
        source = Path(temporary) / "triton_hstu_attention.py"
        source.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.meta_triton),
                    "show",
                    f"{base}:{relative}/triton_hstu_attention.py",
                ]
            )
        )
        name = "dlrm_upstream_hstu_reference"
        spec = importlib.util.spec_from_file_location(name, source)
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[name] = upstream
        spec.loader.exec_module(upstream)
        upstream.configure_autows(upstream.HSTUAutoWSConfig())

        for index in indexes:
            seed, distribution, scale, layout = cases[index]
            cpu = torch.Generator().manual_seed(20261105 + seed * 1009 + args.seq)
            low = max(21, int(args.seq * 0.8)) if seed % 2 == 0 else 21
            lengths = torch.randint(low, args.seq + 1, (args.batch,), generator=cpu)
            boundaries = sorted(
                {
                    21,
                    31,
                    32,
                    33,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    255,
                    256,
                    257,
                    511,
                    512,
                    513,
                    1023,
                    1024,
                    1025,
                    2047,
                    2048,
                    2049,
                    args.seq - 1,
                    args.seq,
                }
            )
            boundaries = [n for n in boundaries if 21 <= n <= args.seq][: args.batch]
            lengths[: len(boundaries)] = torch.tensor(boundaries)
            lengths[-1] = args.seq
            targets_cpu = (
                torch.randint(0, 21, (args.batch,), generator=cpu)
                if seed % 2
                else torch.full((args.batch,), 20)
            )
            offsets_cpu = torch.cat(
                [torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]
            )
            offsets, targets = offsets_cpu.cuda(), targets_cpu.cuda()
            total = int(offsets_cpu[-1])
            rng = torch.Generator(device="cuda").manual_seed(
                810000 + seed * 8191 + args.seq * 7 + index
            )
            packed = torch.empty((total, 4, 384), device="cuda", dtype=torch.bfloat16)
            if distribution == "normal":
                packed.normal_(0, scale, generator=rng)
            elif distribution == "uniform":
                packed.uniform_(-scale, scale, generator=rng)
            else:
                packed.zero_()
            q, k, v = [x.detach().requires_grad_() for x in packed.split(128, -1)]
            if layout == "contiguous":
                q, k, v = [x.detach().contiguous().requires_grad_() for x in (q, k, v)]
            dout = torch.empty(
                (total, 4, 128), device="cuda", dtype=torch.bfloat16
            ).normal_(generator=rng)
            args_op, extra = get_parser().parse_known_args(
                ["--op", "ragged_attention", "--precision", "bf16", "--mode", "fwd"]
            )
            op = operator.Operator(tb_args=args_op, extra_args=extra)
            inputs = (q, k, v, offsets, targets, args.seq, 0.95)
            attn_scale = torch.tensor(
                1.0 / args.seq, device="cuda", dtype=torch.float32
            )
            result = upstream.triton_hstu_mha(
                args.seq,
                1.0 / 128,
                q,
                k,
                v,
                offsets,
                attn_scale,
                num_targets=targets,
                enable_tma=True,
                sort_by_length=True,
            )
            gradients = torch.autograd.grad(result, (q, k, v), dout)
            actual = {
                "upstream": dict(
                    zip(("out", "dq", "dk", "dv"), (result.detach(), *gradients))
                )
            }
            for label, provider in [
                ("hammer", "hstu_triton_hammer"),
                ("hybrid", "hstu_triton_hybrid"),
            ]:
                result = getattr(op, provider)(*inputs)()
                gradients = torch.autograd.grad(result, (q, k, v), dout)
                assert result.dtype == torch.bfloat16
                assert all(gradient.dtype == torch.bfloat16 for gradient in gradients)
                actual[label] = dict(
                    zip(("out", "dq", "dk", "dv"), (result.detach(), *gradients))
                )
            ids = torch.repeat_interleave(
                torch.arange(args.batch, device="cuda"), lengths.cuda()
            )
            record = dict(
                case=index,
                seed=seed,
                distribution=distribution,
                scale=scale,
                layout=layout,
                total_tokens=total,
                direct={},
                fp32={},
                fp32_sequences=[],
            )
            for label in ["hammer", "hybrid"]:
                record["direct"][label] = {
                    field: full_metric(
                        value, actual["upstream"][field], ids, offsets[1:]
                    )
                    for field, value in actual[label].items()
                }
            selected = {0, args.batch - 1, int(lengths.argmin()), int(lengths.argmax())}
            selected.update(
                random.Random(seed * 1009 + args.seq).sample(
                    range(args.batch), min(args.fp32_sequences, args.batch)
                )
            )
            # Explicit regression sequence from the prior B3004096 dQ result.
            if args.seq == 4096 and args.batch > 304:
                selected.add(304)
            record["fp32_sequences"] = sorted(selected)
            record["fp32"] = {label: [] for label in actual}
            for z in sorted(selected):
                begin, end = map(int, offsets_cpu[z : z + 2])
                n, target = end - begin, int(targets_cpu[z])
                qf, kf, vf = [
                    x[begin:end].detach().float().requires_grad_() for x in (q, k, v)
                ]
                rows = torch.arange(n, device="cuda")[:, None]
                cols = torch.arange(n, device="cuda")[None, :]
                valid = (rows == cols) | ((cols < rows) & (cols < n - target))
                score = torch.einsum("qhd,khd->hqk", qf, kf) / 128
                reference = torch.einsum(
                    "hqk,khd->qhd",
                    torch.nn.functional.silu(score) * valid[None] / args.seq,
                    vf,
                )
                ref_gradients = torch.autograd.grad(
                    reference, (qf, kf, vf), dout[begin:end].float()
                )
                expected = dict(
                    zip(("out", "dq", "dk", "dv"), (reference, *ref_gradients))
                )
                for label, values in actual.items():
                    record["fp32"][label].append(
                        dict(
                            sequence=z,
                            length=n,
                            metrics={
                                field: metric(value[begin:end], expected[field])
                                for field, value in values.items()
                            },
                        )
                    )
                del qf, kf, vf, reference, ref_gradients, expected, score, valid
            direct_pass = all(
                m["finite"]
                and m["max_sequence_relative_l2"] < 0.02
                and m["max_last_row_relative_l2"] < 0.03
                for fields in record["direct"].values()
                for m in fields.values()
            )
            fp32_pass = all(
                m["finite"]
                and m["relative_l2"] < 0.01
                and m["last_row_relative_l2"] < 0.01
                for sequence in record["fp32"]["hybrid"]
                for m in sequence["metrics"].values()
            )
            record.update(
                direct_pass=direct_pass,
                hybrid_fp32_pass=fp32_pass,
                passed=direct_pass and fp32_pass,
            )
            records.append(record)
            with (args.output / "checks.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                json.dumps(
                    dict(
                        seq=args.seq,
                        case=index,
                        direct=direct_pass,
                        hybrid_fp32=fp32_pass,
                    )
                ),
                flush=True,
            )
            save(
                args.output / "progress.json",
                dict(
                    complete_cases=len(records),
                    expected_cases=len(indexes),
                    failures=sum(not r["passed"] for r in records),
                    elapsed_seconds=time.time() - started,
                ),
            )
            del actual, values, result, gradients, q, k, v, packed, dout, inputs, ids
            gc.collect()
    summary = dict(
        complete=True,
        cases=len(records),
        failures=sum(not r["passed"] for r in records),
        fp32_sequence_checks=sum(len(r["fp32_sequences"]) for r in records),
        provenance=provenance,
    )
    save(args.output / "summary.json", summary)
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
