from __future__ import annotations

"""
GPT-optimized Digital→Digital (D2D) simulation module.

Goals vs original d2d.py
- Faster waveform generation (vectorized / preallocated arrays; no per-bit np.concatenate).
- Faster sampling/decoding for most schemes (reshape + vectorized means / mid-sample slicing).
- O(n) B8ZS and HDB3 scramblers (no repeated slice+all(...) lookahead checks).
- Keep the same public API and metadata structure so the existing tests still apply.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

# Robust import when this file is imported from comm_sim/GPT_optimized via sys.path injection.
try:
    from utils import SimParams, SimResult, make_time_axis, ensure_even
except ImportError:  # pragma: no cover
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _comm_sim = os.path.dirname(_here)
    if _comm_sim not in sys.path:
        sys.path.insert(0, _comm_sim)
    from utils import SimParams, SimResult, make_time_axis, ensure_even


# =========================
# Internal helpers (fast)
# =========================

def _as_bit_array(bits: List[int]) -> np.ndarray:
    """Convert Python list[0/1] to a compact NumPy array."""
    if not bits:
        return np.empty(0, dtype=np.uint8)
    return np.asarray(bits, dtype=np.uint8)


def _repeat_per_bit(levels: np.ndarray, Ns: int) -> np.ndarray:
    """Repeat one level per bit to Ns samples/bit, returning float wave."""
    if levels.size == 0:
        return np.empty(0, dtype=float)
    return np.repeat(levels.astype(float, copy=False), Ns)


def _mid_samples(wave: np.ndarray, Ns: int) -> np.ndarray:
    """Mid-sample per bit (vectorized)."""
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    nbits = int(len(wave) // Ns)
    if nbits <= 0:
        return np.empty(0, dtype=float)
    mid = Ns // 2
    return wave[mid:nbits * Ns:Ns]


def _sample_bit_levels_ternary(wave: np.ndarray, Ns: int) -> np.ndarray:
    """Return per-bit ternary levels in {-1,0,+1} using the original threshold."""
    mid = _mid_samples(wave, Ns)
    if mid.size == 0:
        return np.empty(0, dtype=np.int8)
    out = np.zeros(mid.shape[0], dtype=np.int8)
    pos = mid > 0.5
    neg = mid < -0.5
    out[pos] = 1
    out[neg] = -1
    return out


# =========================
# Encoding (vectorized)
# =========================

def _nrzl_levels(bits: np.ndarray) -> np.ndarray:
    # Book convention: 0 => +1, 1 => -1
    return (1 - 2 * bits.astype(np.int8, copy=False)).astype(np.int8, copy=False)


def _nrzi_levels(bits: np.ndarray, start_level: int) -> np.ndarray:
    if bits.size == 0:
        return np.empty(0, dtype=np.int8)
    # out[i] = start_level * (-1)^(sum(bits[0:i+1]))
    prefix_ones = np.cumsum(bits, dtype=np.int64)
    parity = (prefix_ones & 1).astype(np.int8, copy=False)          # 0 even, 1 odd
    sign = (1 - 2 * parity).astype(np.int8, copy=False)             # +1 for even, -1 for odd
    return (int(start_level) * sign).astype(np.int8, copy=False)


def _manchester_wave(bits: np.ndarray, Ns: int) -> np.ndarray:
    Ns = ensure_even(int(Ns))
    h = Ns // 2
    n = int(bits.size)
    if n == 0:
        return np.empty(0, dtype=float)

    first = np.where(bits == 1, -1.0, +1.0)          # 1: low->high, 0: high->low
    segs = np.empty((n, Ns), dtype=float)
    segs[:, :h] = first[:, None]
    segs[:, h:] = (-first)[:, None]
    return segs.reshape(-1)


def _diff_manchester_wave(bits: np.ndarray, Ns: int, start_level: float) -> np.ndarray:
    Ns = ensure_even(int(Ns))
    h = Ns // 2
    n = int(bits.size)
    if n == 0:
        return np.empty(0, dtype=float)

    segs = np.empty((n, Ns), dtype=float)
    level = float(start_level)

    # Convention: 0 => transition at start; 1 => no transition at start. Always mid-bit transition.
    for i, b in enumerate(bits.tolist()):  # tolist() is fast for uint8; keeps inner loop tiny
        if b == 0:
            level *= -1.0
        segs[i, :h] = level
        level *= -1.0
        segs[i, h:] = level

    return segs.reshape(-1)


def _ami_levels(bits: np.ndarray, last_pulse_init: int) -> np.ndarray:
    n = int(bits.size)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    levels = np.zeros(n, dtype=np.int8)
    ones = np.flatnonzero(bits)
    if ones.size:
        k = np.arange(1, ones.size + 1, dtype=np.int64)
        # (-1)^k  => odd:-1, even:+1  => 1 - 2*(k&1)
        sign = (1 - 2 * (k & 1)).astype(np.int8, copy=False)
        pulses = (int(last_pulse_init) * sign).astype(np.int8, copy=False)
        levels[ones] = pulses
    return levels


def _pseudoternary_levels(bits: np.ndarray, last_zero_pulse_init: int) -> np.ndarray:
    n = int(bits.size)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    levels = np.zeros(n, dtype=np.int8)  # bit=1 => 0 level, by definition
    zeros = np.flatnonzero(bits == 0)
    if zeros.size:
        k = np.arange(1, zeros.size + 1, dtype=np.int64)
        sign = (1 - 2 * (k & 1)).astype(np.int8, copy=False)
        pulses = (int(last_zero_pulse_init) * sign).astype(np.int8, copy=False)
        levels[zeros] = pulses
    return levels


def _b8zs_scramble_ami(bits: np.ndarray, last_pulse_init: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    AMI + B8ZS scrambling in O(n).
    When 8 consecutive zeros occur, replace the last 8 ternary symbols with:
        000 V B 0 B V
    where V == last_pulse (violation) and B == -V.
    """
    n = int(bits.size)
    out = np.zeros(n, dtype=np.int8)
    meta: Dict[str, Any] = {"substitutions": []}

    last_pulse = int(last_pulse_init)
    run_zeros = 0

    for i, b in enumerate(bits.tolist()):
        if b == 0:
            out[i] = 0
            run_zeros += 1
            if run_zeros == 8:
                v = last_pulse
                B = -v
                pattern = np.array([0, 0, 0, v, B, 0, B, v], dtype=np.int8)
                out[i - 7:i + 1] = pattern
                meta["substitutions"].append({
                    "pos": i - 7,
                    "type": "B8ZS",
                    "pattern": pattern.tolist(),
                    "last_pulse": last_pulse,
                })
                # last_pulse remains v (same as original)
                run_zeros = 0
        else:
            run_zeros = 0
            last_pulse *= -1
            out[i] = last_pulse

    return out, meta


def _hdb3_scramble_ami(
    bits: np.ndarray,
    last_pulse_init: int,
    nonzero_since_violation_init: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    AMI + HDB3 scrambling in O(n), using the original parity rule.

    For any run of four 0s:
      - if nonzero_since_violation is odd:  000V, where V == last_pulse
      - else (even):                        B00V, where B == -last_pulse and V == B (violation)
    """
    n = int(bits.size)
    out = np.zeros(n, dtype=np.int8)
    meta: Dict[str, Any] = {"substitutions": []}

    last_pulse = int(last_pulse_init)
    nonzero_since_violation = int(nonzero_since_violation_init)
    run_zeros = 0

    for i, b in enumerate(bits.tolist()):
        if b == 0:
            out[i] = 0
            run_zeros += 1
            if run_zeros == 4:
                start = i - 3
                if nonzero_since_violation % 2 == 1:
                    # 000V: V has same polarity as last_pulse
                    V = last_pulse
                    pattern = np.array([0, 0, 0, V], dtype=np.int8)
                    out[start:i + 1] = pattern
                    meta["substitutions"].append({
                        "pos": start,
                        "type": "HDB3",
                        "pattern": pattern.tolist(),
                        "rule": "000V",
                        "last_pulse_before": last_pulse,
                        "count_since_last_sub": nonzero_since_violation,
                    })
                    # last_pulse stays V (unchanged)
                else:
                    # B00V: B is normal (opposite last_pulse), V == B (violation repeats B)
                    B = -last_pulse
                    V = B
                    pattern = np.array([B, 0, 0, V], dtype=np.int8)
                    out[start:i + 1] = pattern
                    meta["substitutions"].append({
                        "pos": start,
                        "type": "HDB3",
                        "pattern": pattern.tolist(),
                        "rule": "B00V",
                        "last_pulse_before": last_pulse,
                        "count_since_last_sub": nonzero_since_violation,
                    })
                    last_pulse = V

                nonzero_since_violation = 0
                run_zeros = 0
        else:
            run_zeros = 0
            last_pulse *= -1
            out[i] = last_pulse
            nonzero_since_violation += 1

    return out, meta


# =========================
# Decoding
# =========================

def _decode_manchester(wave: np.ndarray, Ns: int) -> List[int]:
    Ns = ensure_even(int(Ns))
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    nbits = int(len(wave) // Ns)
    if nbits == 0:
        return []
    w = wave[:nbits * Ns].reshape(nbits, Ns)
    h = Ns // 2
    first = w[:, :h].mean(axis=1)
    second = w[:, h:].mean(axis=1)
    # 1: low->high => first < second
    return (first < second).astype(np.int8).tolist()


def _decode_diff_manchester(wave: np.ndarray, Ns: int, start_level: float) -> List[int]:
    Ns = ensure_even(int(Ns))
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    nbits = int(len(wave) // Ns)
    if nbits == 0:
        return []

    w = wave[:nbits * Ns].reshape(nbits, Ns)
    h = Ns // 2
    first = w[:, :h].mean(axis=1)
    second = w[:, h:].mean(axis=1)

    bits: List[int] = []
    prev_last = 1.0 if float(start_level) >= 0 else -1.0
    for i in range(nbits):
        start_transition = (np.sign(first[i]) != np.sign(prev_last))
        bits.append(0 if start_transition else 1)
        prev_last = second[i]
    return bits


def _descramble_b8zs(tern: List[int], last_pulse_init: int) -> Tuple[List[int], Dict[str, Any]]:
    out_bits: List[int] = []
    meta: Dict[str, Any] = {"descramble_hits": []}
    last_pulse = int(last_pulse_init)
    i = 0
    n = len(tern)

    while i < n:
        if i + 8 <= n:
            chunk = tern[i:i + 8]
            # Candidate must look like 000 v b 0 b v
            if chunk[0:3] == [0, 0, 0] and chunk[5] == 0:
                v = chunk[3]
                b = chunk[4]
                if v in (+1, -1) and b in (+1, -1):
                    if chunk == [0, 0, 0, v, b, 0, b, v] and b == -v and last_pulse == v:
                        out_bits.extend([0] * 8)
                        meta["descramble_hits"].append({
                            "pos": i,
                            "type": "B8ZS",
                            "chunk": chunk,
                            "v": v,
                            "b": b,
                            "last_pulse": last_pulse,
                        })
                        # last_pulse remains v
                        i += 8
                        continue

        val = tern[i]
        if val == 0:
            out_bits.append(0)
        else:
            out_bits.append(1)
            last_pulse = val
        i += 1

    return out_bits, meta


def _descramble_hdb3(tern: List[int], last_nonzero_init: int) -> Tuple[List[int], Dict[str, Any]]:
    out_bits: List[int] = []
    meta: Dict[str, Any] = {"descramble_hits": []}
    last_nonzero: int | None = int(last_nonzero_init) if last_nonzero_init is not None else None

    def append_from_val(v: int) -> None:
        nonlocal last_nonzero
        if v == 0:
            out_bits.append(0)
        else:
            out_bits.append(1)
            last_nonzero = v

    i = 0
    n = len(tern)
    while i < n:
        if i + 4 <= n:
            w = tern[i:i + 4]

            # 000V: [0,0,0,V] where V repeats last_nonzero (violation)
            if w[0] == 0 and w[1] == 0 and w[2] == 0 and w[3] in (+1, -1):
                V = w[3]
                if last_nonzero is not None and V == last_nonzero:
                    out_bits.extend([0, 0, 0, 0])
                    meta["descramble_hits"].append({
                        "pos": i,
                        "type": "HDB3",
                        "rule": "000V",
                        "window": w,
                        "last_nonzero": last_nonzero,
                    })
                    last_nonzero = V
                    i += 4
                    continue

            # B00V: [B,0,0,V] where B == -last_nonzero and V == B
            if w[0] in (+1, -1) and w[1] == 0 and w[2] == 0 and w[3] in (+1, -1):
                B, V = w[0], w[3]
                if last_nonzero is not None and B == -last_nonzero and V == B:
                    out_bits.extend([0, 0, 0, 0])
                    meta["descramble_hits"].append({
                        "pos": i,
                        "type": "HDB3",
                        "rule": "B00V",
                        "window": w,
                        "last_nonzero": last_nonzero,
                    })
                    last_nonzero = V
                    i += 4
                    continue

        append_from_val(tern[i])
        i += 1

    return out_bits, meta


# =========================
# Public API (same as d2d.py)
# =========================

def line_encode(
    bits: List[int],
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,
    hdb3_nonzero_since_violation_init: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bits_np = _as_bit_array(bits)
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}

    if scheme == "NRZ-L":
        levels = _nrzl_levels(bits_np)
        return _repeat_per_bit(levels, Ns), meta

    if scheme == "NRZI":
        levels = _nrzi_levels(bits_np, start_level=nrzi_start_level)
        return _repeat_per_bit(levels, Ns), meta

    if scheme == "Manchester":
        Ns_adj = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns_adj
        return _manchester_wave(bits_np, Ns_adj), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns_adj = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns_adj
        return _diff_manchester_wave(bits_np, Ns_adj, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _ami_levels(bits_np, last_pulse_init=last_pulse_init)
        return _repeat_per_bit(levels, Ns), meta

    if scheme == "Pseudoternary":
        levels = _pseudoternary_levels(bits_np, last_zero_pulse_init=last_zero_pulse_init)
        return _repeat_per_bit(levels, Ns), meta

    if scheme == "B8ZS":
        levels, smeta = _b8zs_scramble_ami(bits_np, last_pulse_init=last_pulse_init)
        meta.update(smeta)
        return _repeat_per_bit(levels, Ns), meta

    if scheme == "HDB3":
        levels, smeta = _hdb3_scramble_ami(
            bits_np,
            last_pulse_init=last_pulse_init,
            nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
        )
        meta.update(smeta)
        return _repeat_per_bit(levels, Ns), meta

    raise ValueError(f"Unknown scheme: {scheme}")


def line_decode(
    wave: np.ndarray,
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,  # unused in decode; kept for API symmetry
) -> Tuple[List[int], Dict[str, Any]]:
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}

    if scheme == "NRZ-L":
        mid = _mid_samples(wave, Ns)
        # Book convention: + => bit 0, - => bit 1
        return (mid < 0).astype(np.int8).tolist(), meta

    if scheme == "NRZI":
        levels = _sample_bit_levels_ternary(wave, Ns)  # -1/+1 expected (no zeros)
        bits_out: List[int] = []
        prev = int(nrzi_start_level)
        for v in levels.tolist():
            bits_out.append(1 if v != prev else 0)
            prev = v
        return bits_out, meta

    if scheme == "Manchester":
        Ns_adj = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns_adj
        return _decode_manchester(wave, Ns_adj), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns_adj = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns_adj
        return _decode_diff_manchester(wave, Ns_adj, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _sample_bit_levels_ternary(wave, Ns)
        return (levels != 0).astype(np.int8).tolist(), meta

    if scheme == "Pseudoternary":
        levels = _sample_bit_levels_ternary(wave, Ns)
        return (levels == 0).astype(np.int8).tolist(), meta

    if scheme == "B8ZS":
        tern = _sample_bit_levels_ternary(wave, Ns).tolist()
        bits_out, dmeta = _descramble_b8zs(tern, last_pulse_init=last_pulse_init)
        meta.update(dmeta)
        return bits_out, meta

    if scheme == "HDB3":
        tern = _sample_bit_levels_ternary(wave, Ns).tolist()
        bits_out, dmeta = _descramble_hdb3(tern, last_nonzero_init=last_pulse_init)
        meta.update(dmeta)
        return bits_out, meta

    raise ValueError(f"Unknown scheme: {scheme}")


def simulate_d2d(
    bits: List[int],
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,
    hdb3_nonzero_since_violation_init: int = 0,
) -> SimResult:
    tx, meta_tx = line_encode(
        bits, scheme, params,
        nrzi_start_level=nrzi_start_level,
        diff_start_level=diff_start_level,
        last_pulse_init=last_pulse_init,
        last_zero_pulse_init=last_zero_pulse_init,
        hdb3_nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
    )
    decoded, meta_rx = line_decode(
        tx, scheme, params,
        nrzi_start_level=nrzi_start_level,
        diff_start_level=diff_start_level,
        last_pulse_init=last_pulse_init,
        last_zero_pulse_init=last_zero_pulse_init,
    )

    t = make_time_axis(len(tx), params.fs)
    meta = {
        "scheme": scheme,
        "encode": meta_tx,
        "decode": meta_rx,
        "match": (decoded == bits),
        "input_len": len(bits),
    }
    return SimResult(
        t=t,
        signals={"tx": tx},
        bits={"input": bits, "decoded": decoded},
        meta=meta,
    )
