#!/usr/bin/env python3
"""Shared INT8 contract for the Phase 1 BasicBlock flow.

The contract is intentionally small and explicit so the Python export step,
the C++ checker, and the RTL testbench can all implement the same math:

- Activations are signed INT8 with per-channel scales.
- Weights are signed INT8. Each output channel has its own accumulator scale,
  and the input activation scales are folded into the offline weight packing.
- Bias is stored as signed INT32 in the accumulator domain.
- Convolution accumulation is signed INT32.
- Requantization uses an integer multiplier with a fixed right shift.
- LeakyReLU(0.1) is implemented as a signed integer multiply by 3277 / 2^15.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

INT8_MIN = -128
INT8_MAX = 127
INT8_QMAX = 127
ACC_WIDTH_BITS = 32
REQUANT_SHIFT_BITS = 24
LEAKY_RELU_NUM = 3277
LEAKY_RELU_SHIFT = 15
EPSILON = 1e-8


@dataclass(frozen=True)
class FixedMultiplier:
    multiplier: int
    shift: int = REQUANT_SHIFT_BITS


def _reshape_channel_vector(values: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return values.reshape(shape)


def clamp_int8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, INT8_MIN, INT8_MAX).astype(np.int8)


def round_shift_signed(values: np.ndarray, shift: int) -> np.ndarray:
    """Round half away from zero before arithmetic right shift."""
    values = values.astype(np.int64, copy=False)
    offset = 1 << (shift - 1)
    pos = (values + offset) >> shift
    neg = -(((-values) + offset) >> shift)
    return np.where(values >= 0, pos, neg).astype(np.int64)


def symmetric_channel_scales(
    tensor: np.ndarray,
    axis: int = 1,
    qmax: int = INT8_QMAX,
) -> np.ndarray:
    dims = tuple(d for d in range(tensor.ndim) if d != axis)
    max_abs = np.max(np.abs(tensor), axis=dims)
    return np.maximum(max_abs / float(qmax), EPSILON).astype(np.float64)


def quantize_activation_per_channel(
    tensor: np.ndarray,
    scales: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    scale_view = _reshape_channel_vector(scales.astype(np.float64), axis, tensor.ndim)
    quantized = np.rint(tensor / scale_view)
    return clamp_int8(quantized)


def dequantize_activation_per_channel(
    tensor_q: np.ndarray,
    scales: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    scale_view = _reshape_channel_vector(scales.astype(np.float64), axis, tensor_q.ndim)
    return tensor_q.astype(np.float32) * scale_view.astype(np.float32)


def choose_weight_acc_scales(weight: np.ndarray, input_scales: np.ndarray) -> np.ndarray:
    """Pick one accumulator-domain scale per output channel.

    The input activation channel scales are folded into the offline weight
    quantization, so the runtime MAC uses only integer activations and weights.
    """

    assert weight.ndim == 4, weight.shape
    assert input_scales.ndim == 1, input_scales.shape
    scaled = np.abs(weight.astype(np.float64)) * input_scales.reshape(1, -1, 1, 1)
    max_abs = np.max(scaled, axis=(1, 2, 3))
    return np.maximum(max_abs / float(INT8_QMAX), EPSILON).astype(np.float64)


def quantize_weight_with_input_scales(
    weight: np.ndarray,
    input_scales: np.ndarray,
    acc_scales: np.ndarray,
) -> np.ndarray:
    scaled = weight.astype(np.float64) * input_scales.reshape(1, -1, 1, 1)
    scaled /= acc_scales.reshape(-1, 1, 1, 1)
    quantized = np.rint(scaled)
    return clamp_int8(quantized)


def quantize_bias_to_acc_domain(bias: np.ndarray, acc_scales: np.ndarray) -> np.ndarray:
    assert bias.ndim == 1, bias.shape
    quantized = np.rint(bias.astype(np.float64) / acc_scales.astype(np.float64))
    return quantized.astype(np.int32)


def integer_multiplier(real_scale: np.ndarray | float, shift: int = REQUANT_SHIFT_BITS) -> np.ndarray:
    scaled = np.rint(np.asarray(real_scale, dtype=np.float64) * (1 << shift))
    return scaled.astype(np.int64)


def apply_integer_multiplier(values: np.ndarray, multiplier: np.ndarray, axis: int = 1) -> np.ndarray:
    mult_view = _reshape_channel_vector(np.asarray(multiplier, dtype=np.int64), axis, values.ndim)
    products = values.astype(np.int64) * mult_view
    return round_shift_signed(products, REQUANT_SHIFT_BITS)


def leaky_relu_int32(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.int64, copy=False)
    neg = round_shift_signed(values * LEAKY_RELU_NUM, LEAKY_RELU_SHIFT)
    return np.where(values >= 0, values, neg).astype(np.int32)


def requantize_accumulator(values: np.ndarray, multiplier: np.ndarray, axis: int = 1) -> np.ndarray:
    scaled = apply_integer_multiplier(values.astype(np.int64), multiplier, axis=axis)
    return clamp_int8(scaled)


def rescale_int8_to_int32(values: np.ndarray, multiplier: np.ndarray, axis: int = 1) -> np.ndarray:
    return apply_integer_multiplier(values.astype(np.int64), multiplier, axis=axis).astype(np.int32)


def twos_complement_hex(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{(int(value) & mask):0{bits // 4}x}"


def write_memh(path: Path, values: Iterable[int], bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for value in values:
            handle.write(twos_complement_hex(int(value), bits))
            handle.write("\n")


def write_ndarray_memh(path: Path, array: np.ndarray, bits: int) -> None:
    write_memh(path, array.reshape(-1), bits)


def per_channel_absmax_summary(tensor: np.ndarray, axis: int = 1) -> dict[str, float]:
    dims = tuple(d for d in range(tensor.ndim) if d != axis)
    max_abs = np.max(np.abs(tensor), axis=dims).astype(np.float64)
    return {
        "min": float(np.min(max_abs)),
        "max": float(np.max(max_abs)),
        "mean": float(np.mean(max_abs)),
    }
