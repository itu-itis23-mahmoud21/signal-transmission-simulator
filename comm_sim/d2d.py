from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from utils import SimParams, SimResult, make_time_axis, ensure_even

# ---------- Encoding helpers (bit-level ternary) ----------

def _nrzl_levels(bits: List[int]) -> List[int]:
    # Book convention: 0 = high (+1), 1 = low (-1)
    return [ -1 if b == 1 else +1 for b in bits]


def _nrzi_levels(bits: List[int], start_level: int = -1) -> List[int]:
    level = start_level
    out = []
    for b in bits:
        if b == 1:
            level *= -1  # transition at start
        out.append(level)
    return out


def _manchester_samples(bits: List[int], Ns: int) -> np.ndarray:
    Ns = ensure_even(Ns)
    h = Ns // 2
    # IEEE-style: 1 = low->high, 0 = high->low
    chunks = []
    for b in bits:
        if b == 1:
            first, second = -1.0, +1.0
        else:
            first, second = +1.0, -1.0
        chunks.append(np.concatenate([np.full(h, first), np.full(h, second)]))
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def _diff_manchester_samples(bits: List[int], Ns: int, start_level: float = +1.0) -> np.ndarray:
    Ns = ensure_even(Ns)
    h = Ns // 2
    level = start_level
    chunks = []
    # Convention: 0 => transition at start; 1 => no transition at start.
    for b in bits:
        if b == 0:
            level *= -1
        first_half = np.full(h, level)
        level *= -1  # always mid-bit transition
        second_half = np.full(h, level)
        chunks.append(np.concatenate([first_half, second_half]))
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def _ami_levels(bits: List[int], last_pulse_init: int = -1) -> List[int]:
    last = last_pulse_init
    out = []
    for b in bits:
        if b == 0:
            out.append(0)
        else:
            last *= -1
            out.append(last)
    return out


def _pseudoternary_levels(bits: List[int], last_zero_pulse_init: int = -1) -> List[int]:
    # Pseudoternary: 0 = alternating +/- (successive zeros), 1 = 0 level
    # last_zero_pulse_init represents the polarity of the most recent preceding 0 (before the sequence starts)
    last = last_zero_pulse_init
    out = []
    for b in bits:
        if b == 1:
            out.append(0)
        else:
            last *= -1
            out.append(last)
    return out


def _b8zs_scramble_ami(bits: List[int], last_pulse_init: int = -1) -> Tuple[List[int], Dict[str, Any]]:
    out = []
    meta = {"substitutions": []}
    last_pulse = last_pulse_init
    i = 0
    while i < len(bits):
        if i + 8 <= len(bits) and all(b == 0 for b in bits[i:i+8]):
            v = last_pulse
            b = -v
            pattern = [0, 0, 0, v, b, 0, b, v]
            out.extend(pattern)
            meta["substitutions"].append({"pos": i, "type": "B8ZS", "pattern": pattern, "last_pulse": last_pulse})
            # last nonzero in pattern is v which equals last_pulse
            i += 8
            continue

        bit = bits[i]
        if bit == 0:
            out.append(0)
        else:
            last_pulse *= -1
            out.append(last_pulse)
        i += 1
    return out, meta


def _hdb3_scramble_ami(
    bits: List[int],
    last_pulse_init: int = -1,
    nonzero_since_violation_init: int = 0,   # 0=even, 1=odd
) -> Tuple[List[int], Dict[str, Any]]:
    out: List[int] = []
    meta = {"substitutions": []}
    last_pulse = last_pulse_init
    nonzero_since_violation = int(nonzero_since_violation_init)

    i = 0
    while i < len(bits):
        if i + 4 <= len(bits) and all(b == 0 for b in bits[i:i+4]):
            if nonzero_since_violation % 2 == 1:
                # 000V, V has same polarity as last_pulse (violation vs last nonzero)
                v = last_pulse
                pattern = [0, 0, 0, v]
                out.extend(pattern)
                meta["substitutions"].append({
                    "pos": i,
                    "type": "HDB3",
                    "pattern": pattern,
                    "rule": "000V",
                    "last_pulse_before": last_pulse,
                    "count_since_last_sub": nonzero_since_violation
                })
            else:
                # B00V: B is normal (opposite of last_pulse), and V == B (violation: repeats B)
                B = -last_pulse
                V = B
                pattern = [B, 0, 0, V]
                out.extend(pattern)
                meta["substitutions"].append({
                    "pos": i,
                    "type": "HDB3",
                    "pattern": pattern,
                    "rule": "B00V",
                    "last_pulse_before": last_pulse,
                    "count_since_last_sub": nonzero_since_violation
                })
            # After violation, reset counter; last_pulse becomes polarity of V (which equals last_pulse for 000V; equals B for B00V)
            last_pulse = out[-1] if out[-1] != 0 else last_pulse
            nonzero_since_violation = 0
            i += 4
            continue

        bit = bits[i]
        if bit == 0:
            out.append(0)
        else:
            last_pulse *= -1
            out.append(last_pulse)
            nonzero_since_violation += 1
        i += 1

    return out, meta


def _ternary_to_wave(levels: List[int], Ns: int) -> np.ndarray:
    if not levels:
        return np.array([], dtype=float)
    return np.repeat(np.array(levels, dtype=float), Ns)


# ---------- Public API ----------

def line_encode(
    bits: List[int],
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,       # for AMI/B8ZS/HDB3 (preceding '1' polarity)
    last_zero_pulse_init: int = -1,  # for Pseudoternary (preceding '0' polarity)
    hdb3_nonzero_since_violation_init: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}

    if scheme == "NRZ-L":
        levels = _nrzl_levels(bits)
        return _ternary_to_wave(levels, Ns), meta

    if scheme == "NRZI":
        levels = _nrzi_levels(bits, start_level=nrzi_start_level)
        return _ternary_to_wave(levels, Ns), meta

    if scheme == "Manchester":
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _manchester_samples(bits, Ns), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _diff_manchester_samples(bits, Ns, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _ami_levels(bits, last_pulse_init=last_pulse_init)
        return _ternary_to_wave(levels, Ns), meta

    if scheme == "Pseudoternary":
        levels = _pseudoternary_levels(bits, last_zero_pulse_init=last_zero_pulse_init)
        return _ternary_to_wave(levels, Ns), meta

    if scheme == "B8ZS":
        levels, smeta = _b8zs_scramble_ami(bits, last_pulse_init=last_pulse_init)
        meta.update(smeta)
        return _ternary_to_wave(levels, Ns), meta

    if scheme == "HDB3":
        levels, smeta = _hdb3_scramble_ami(
            bits,
            last_pulse_init=last_pulse_init,
            nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
        )
        meta.update(smeta)
        return _ternary_to_wave(levels, Ns), meta

    raise ValueError(f"Unknown scheme: {scheme}")


def _sample_bit_levels(wave: np.ndarray, Ns: int) -> List[int]:
    # Take mid-sample per bit interval
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    nbits = len(wave) // Ns
    out: List[int] = []
    for i in range(nbits):
        seg = wave[i*Ns:(i+1)*Ns]
        mid = seg[len(seg)//2]
        if abs(mid) < 0.5:
            out.append(0)
        else:
            out.append(1 if mid > 0 else -1)
    return out


def _decode_manchester(wave: np.ndarray, Ns: int) -> List[int]:
    Ns = ensure_even(Ns)
    nbits = len(wave) // Ns
    bits: List[int] = []
    h = Ns // 2
    for i in range(nbits):
        seg = wave[i*Ns:(i+1)*Ns]
        first = np.mean(seg[:h])
        second = np.mean(seg[h:])
        # 1: low->high => first < second
        bits.append(1 if first < second else 0)
    return bits


def _decode_diff_manchester(wave: np.ndarray, Ns: int, start_level: float = +1.0) -> List[int]:
    Ns = ensure_even(Ns)
    nbits = len(wave) // Ns
    bits: List[int] = []
    h = Ns // 2
    prev_last = start_level
    for i in range(nbits):
        seg = wave[i*Ns:(i+1)*Ns]
        first_level = np.mean(seg[:h])
        # transition at start?
        start_transition = (np.sign(first_level) != np.sign(prev_last))
        # Convention: 0 => start transition; 1 => no start transition
        bits.append(0 if start_transition else 1)
        prev_last = np.mean(seg[h:])  # last half level
    return bits


def _descramble_b8zs(tern: List[int], last_pulse_init: int = -1) -> Tuple[List[int], Dict[str, Any]]:
    out_bits: List[int] = []
    meta = {"descramble_hits": []}
    last_pulse = last_pulse_init
    i = 0
    while i < len(tern):
        if i + 8 <= len(tern):
            chunk = tern[i:i+8]
            # Candidate must look like 000 v b 0 b v
            if chunk[0:3] == [0, 0, 0] and chunk[5] == 0:
                v = chunk[3]
                b = chunk[4]
                if v in (+1, -1) and b in (+1, -1):
                    if chunk == [0, 0, 0, v, b, 0, b, v] and b == -v and last_pulse == v:
                        out_bits.extend([0]*8)
                        meta["descramble_hits"].append({
                            "pos": i,
                            "type": "B8ZS",
                            "chunk": chunk,          # the 8-symbol ternary pattern seen
                            "v": v,
                            "b": b,
                            "last_pulse": last_pulse
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


def _descramble_hdb3(tern: List[int], last_nonzero_init: int | None = None) -> Tuple[List[int], Dict[str, Any]]:
    out_bits = []
    meta = {"descramble_hits": []}
    last_nonzero = last_nonzero_init

    def append_from_val(v: int):
        nonlocal last_nonzero
        if v == 0:
            out_bits.append(0)
        else:
            out_bits.append(1)
            last_nonzero = v

    i = 0
    while i < len(tern):
        # Lookahead only if we can see a 4-length window ending at i+3
        if i + 4 <= len(tern):
            w = tern[i:i+4]
            # Check 000V: [0,0,0,V] where V is violation vs last_nonzero
            if w[0] == 0 and w[1] == 0 and w[2] == 0 and w[3] in (+1, -1):
                V = w[3]
                if last_nonzero is not None and V == last_nonzero:
                    out_bits.extend([0, 0, 0, 0])
                    meta["descramble_hits"].append({
                        "pos": i,
                        "type": "HDB3",
                        "rule": "000V",
                        "window": w,
                        "last_nonzero": last_nonzero
                    })
                    # IMPORTANT: keep decoder state aligned with received stream
                    last_nonzero = V
                    i += 4
                    continue

            # Check B00V: [B,0,0,V] where B == -last_nonzero and V == B (violation repeats B)
            if w[0] in (+1, -1) and w[1] == 0 and w[2] == 0 and w[3] in (+1, -1):
                B, V = w[0], w[3]
                if last_nonzero is not None and B == -last_nonzero and V == B:
                    out_bits.extend([0, 0, 0, 0])
                    meta["descramble_hits"].append({
                        "pos": i,
                        "type": "HDB3",
                        "rule": "B00V",
                        "window": w,
                        "last_nonzero": last_nonzero
                    })
                    # IMPORTANT: window ends with a nonzero pulse; update state
                    last_nonzero = V
                    i += 4
                    continue

        append_from_val(tern[i])
        i += 1

    return out_bits, meta

def line_decode(
    wave: np.ndarray,
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,  # unused in decode; for API symmetry
) -> Tuple[List[int], Dict[str, Any]]:
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}

    if scheme == "NRZ-L":
        levels = _sample_bit_levels(wave, Ns)
        # Book convention: + => bit 0, - => bit 1
        bits = [0 if v > 0 else 1 for v in levels]
        return bits, meta

    if scheme == "NRZI":
        levels = _sample_bit_levels(wave, Ns)
        # levels are +/-1 (no zeros expected)
        bits: List[int] = []
        prev = nrzi_start_level
        for v in levels:
            bits.append(1 if v != prev else 0)
            prev = v
        return bits, meta

    if scheme == "Manchester":
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _decode_manchester(wave, Ns), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _decode_diff_manchester(wave, Ns, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _sample_bit_levels(wave, Ns)
        bits = [1 if v != 0 else 0 for v in levels]
        return bits, meta

    if scheme == "Pseudoternary":
        levels = _sample_bit_levels(wave, Ns)
        bits = [0 if v != 0 else 1 for v in levels]
        return bits, meta

    if scheme == "B8ZS":
        tern = _sample_bit_levels(wave, Ns)
        bits, dmeta = _descramble_b8zs(tern, last_pulse_init=last_pulse_init)
        meta.update(dmeta)
        return bits, meta

    if scheme == "HDB3":
        tern = _sample_bit_levels(wave, Ns)
        bits, dmeta = _descramble_hdb3(tern, last_nonzero_init=last_pulse_init)
        meta.update(dmeta)
        return bits, meta

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

    n = len(tx)
    t = make_time_axis(n, params.fs)

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
        meta=meta
    )
