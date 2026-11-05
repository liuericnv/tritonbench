"""Exercise the real Hammer/hybrid providers and their configuration isolation."""

import dataclasses
import os

import pytest
import torch
import triton

from tritonbench.operators.ragged_attention import operator
from tritonbench.utils.parser import get_parser


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Blackwell HSTU integration",
)


@pytest.fixture(autouse=True)
def preserve_configuration():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    configure = getattr(operator, "hstu_self_configure", None)
    previous_config = configure() if configure else None
    names = [
        "TRITON_USE_META_WS",
        "TRITON_DISABLE_WSBARRIER_REORDER",
        "TRITON_WS_SMEM_PLAN_SEARCH",
        "TRITON_WS_TMA_REDUCE_STAGING_COPIES",
    ]
    previous_env = {name: os.environ.get(name) for name in names}
    with triton.knobs.nvidia.scope():
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            if configure:
                configure(previous_config)
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def make_operator():
    args, extra = get_parser().parse_known_args(
        ["--op", "ragged_attention", "--precision", "bf16", "--mode", "fwd"]
    )
    return operator.Operator(tb_args=args, extra_args=extra)


def assert_l2(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
    assert float(error) < 0.01, float(error)


@pytest.mark.parametrize("dimension,value_dimension", [(64, 64), (64, 128), (32, 64)])
def test_hstu_hybrid_forward_gradients_and_hammer_isolation(dimension, value_dimension):
    # Removing the provider or sharing its configuration with Hammer must fail.
    assert callable(getattr(operator.Operator, "hstu_triton_hybrid", None))
    torch.backends.cuda.matmul.allow_tf32 = False
    op = make_operator()
    lengths, targets = [65, 129, 257], [0, 7, 20]
    maximum, total, heads = max(lengths), sum(lengths), 2
    offsets = torch.tensor([0, 65, 194, total], device="cuda", dtype=torch.int32)
    target_tensor = torch.tensor(targets, device="cuda", dtype=torch.int32)
    rng = torch.Generator(device="cuda").manual_seed(1761)
    packed = torch.empty(
        (total, heads, 2 * dimension + value_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(0, 0.5, generator=rng)
    q, k, v = [
        tensor.detach().requires_grad_()
        for tensor in packed.split([dimension, dimension, value_dimension], -1)
    ]
    # The operator constructor defaults to D128. Supply the ordinary per-case
    # scale for these shapes rather than depending on that constructor default.
    op.alpha = 1.0 / dimension
    inputs = (q, k, v, offsets, target_tensor, maximum, 0.95)
    hammer = op.hstu_triton_hammer(*inputs)
    baseline_configuration = dataclasses.asdict(operator.hstu_self_configure())
    hybrid = op.hstu_triton_hybrid(*inputs)
    assert dataclasses.asdict(operator.hstu_self_configure()) == baseline_configuration

    qf, kf, vf = [tensor.detach().float().requires_grad_() for tensor in (q, k, v)]
    pieces = []
    start = 0
    for length, target in zip(lengths, targets):
        rows = torch.arange(length, device="cuda")[:, None]
        columns = torch.arange(length, device="cuda")[None, :]
        mask = (rows == columns) | ((columns < rows) & (columns < length - target))
        score = (
            torch.einsum(
                "qhd,khd->hqk", qf[start : start + length], kf[start : start + length]
            )
            / dimension
        )
        weight = torch.nn.functional.silu(score) * mask[None] / maximum
        pieces.append(torch.einsum("hqk,khd->qhd", weight, vf[start : start + length]))
        start += length
    reference = torch.cat(pieces)
    upstream = torch.empty_like(reference, dtype=torch.bfloat16).normal_(generator=rng)
    reference_gradients = torch.autograd.grad(reference, (qf, kf, vf), upstream.float())

    # Keep both graphs alive and run backward in the opposite order, exercising
    # independent callback configuration rather than one-provider-only processes.
    hammer_output, hybrid_output = hammer(), hybrid()
    hybrid_gradients = torch.autograd.grad(hybrid_output, (q, k, v), upstream)
    hammer_gradients = torch.autograd.grad(hammer_output, (q, k, v), upstream)
    for output, gradients in [
        (hammer_output, hammer_gradients),
        (hybrid_output, hybrid_gradients),
    ]:
        assert output.dtype == torch.bfloat16
        start = 0
        for length in lengths:
            assert_l2(output[start : start + length], reference[start : start + length])
            for actual, expected in zip(gradients, reference_gradients):
                assert actual.dtype == torch.bfloat16
                assert_l2(
                    actual[start : start + length], expected[start : start + length]
                )
            start += length
    assert dataclasses.asdict(operator.hstu_self_configure()) == baseline_configuration
