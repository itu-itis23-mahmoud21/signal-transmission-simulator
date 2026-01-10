from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import sys

import numpy as np

# Robust import: allow running from tests that inject different sys.path roots
try:
    from utils import SimParams, SimResult, make_time_axis
except Exception:  # pragma: no cover
    _here = os.path.dirname(os.path.abspath(__file__))
    _proj = os.path.dirname(_here)  # ../comm_sim
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    from utils import SimParams, SimResult, make_time_axis


# =========================
# Helpers (kept compatible)
# =========================

def _validate_bits(bits: List[int]) -> None:
    if bits is None or len(bits) == 0:
        raise ValueError("bits must be a non-empty list of 0/1 integers")
    bad = [b for b in bits if b not in (0, 1)]
    if bad:
        raise ValueError("bits must contain only 0 and 1")


def _pad_bits(bits: List[int], multiple: int) -> Tuple[List[int], int]:
    if multiple <= 1:
        return list(bits), 0
    pad = (-len(bits)) % multiple
    if pad:
        return list(bits) + [0] * pad, pad
    return list(bits), 0


def _warn_params(params: SimParams, extra_freqs: List[float]) -> List[str]:
    warnings: List[str] = []
    fs = float(params.fs)
    nyq = fs / 2.0
    all_freqs = [float(params.fc)] + [float(x) for x in extra_freqs]
    for f in all_freqs:
        if f <= 0:
            warnings.append(f"Frequency {f:.3g} Hz is non-positive; results may be invalid.")
        if f >= nyq:
            warnings.append(f"Frequency {f:.3g} Hz >= Nyquist ({nyq:.3g} Hz): aliasing likely.")
    if params.samples_per_bit <= 2:
        warnings.append("samples_per_bit is very small; demod decisions may be unstable.")
    return warnings


def _ang_dist(a: np.ndarray, b: float) -> np.ndarray:
    # Vectorized angular distance on circle
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _segment_view(x: np.ndarray, seg_len: int) -> np.ndarray:
    # Trim to a whole number of segments then reshape (copy-free)
    nseg = x.size // seg_len
    if nseg <= 0:
        return x[:0].reshape(0, seg_len)
    return x[: nseg * seg_len].reshape(nseg, seg_len)


# =========================
# Mapping tables (compat)
# =========================

# QPSK mapping used in d2a.py (Gray) — kept for readability/compat, even though vectorized logic is used.
_QPSK_MAP: Dict[Tuple[int, int], Tuple[int, int]] = {
    (1, 1): (+1, +1),
    (0, 1): (-1, +1),
    (0, 0): (-1, -1),
    (1, 0): (+1, -1),
}

# For QAM
_QAM_AXIS_BITS_TO_LEVEL_2 = {(0,): -1.0, (1,): +1.0}
_QAM_AXIS_BITS_TO_LEVEL_4 = {
    (0, 0): -3.0,
    (0, 1): -1.0,
    (1, 1): +1.0,
    (1, 0): +3.0,
}

# 16QAM legacy
_16QAM_BITS_TO_LEVEL = _QAM_AXIS_BITS_TO_LEVEL_4
_16QAM_LEVELS = np.array([-3.0, -1.0, +1.0, +3.0], dtype=float)


def _nearest_16qam_level_vec(x: np.ndarray) -> np.ndarray:
    # Piecewise thresholds at -2, 0, 2
    return np.where(
        x < -2.0,
        -3.0,
        np.where(x < 0.0, -1.0, np.where(x < 2.0, +1.0, +3.0)),
    )


# ============================================================
# Core API: modulate(bits, scheme, params, **kwargs) -> (s,meta)
# ============================================================

def modulate(bits: List[int], scheme: str, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    _validate_bits(bits)

    scheme_u = scheme.upper().strip()
    Ns = int(params.samples_per_bit)
    Tb = float(params.Tb)
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)

    meta: Dict[str, Any] = {"scheme": scheme_u}
    warnings = _warn_params(params, extra_freqs=[])

    # Common time base utilities
    # omega_base = 2*pi*t  (useful for constant-frequency carriers)
    # Note: make_time_axis returns seconds, starting at 0, length N
    if scheme_u == "ASK":
        A0 = float(kwargs.get("A0", 0.5))
        A1 = float(kwargs.get("A1", 1.0))
        if A1 <= A0:
            warnings.append("ASK: A1 <= A0; thresholding may be ambiguous.")

        bits_np = np.asarray(bits, dtype=int)
        amps_bit = np.where(bits_np == 1, A1, A0).astype(float)
        N = bits_np.size * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        carrier = np.cos(fc * omega)

        amps_samp = np.repeat(amps_bit, Ns)
        s = Ac * amps_samp * carrier

        meta.update(
            {"A0": A0, "A1": A1, "A_used": amps_bit.tolist(), "warnings": warnings}
        )
        return s.astype(float, copy=False), meta

    if scheme_u == "BFSK":
        tone_sep = float(kwargs.get("tone_sep", 4.0))
        f0 = float(kwargs.get("f0", fc - tone_sep / Tb))
        f1 = float(kwargs.get("f1", fc + tone_sep / Tb))

        if f1 < f0:
            f0, f1 = f1, f0  # keep consistent ordering like d2a.py
        warnings = _warn_params(params, extra_freqs=[f0, f1])

        bits_np = np.asarray(bits, dtype=int)
        N = bits_np.size * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        c0 = np.cos(f0 * omega)
        c1 = np.cos(f1 * omega)

        mask = np.repeat(bits_np == 1, Ns)
        s = Ac * np.where(mask, c1, c0)

        f_used = np.where(bits_np == 1, f1, f0).astype(float).tolist()
        meta.update(
            {
                "tone_sep": tone_sep,
                "f0": f0,
                "f1": f1,
                "f_used": f_used,
                "warnings": warnings,
            }
        )
        return s.astype(float, copy=False), meta

    if scheme_u == "MFSK":
        L = int(kwargs.get("L", 2))
        if L <= 0:
            raise ValueError("MFSK: L must be >= 1")
        M = 2**L
        fd = float(kwargs.get("fd", 2.0))

        bits_p, pad = _pad_bits(bits, L)
        bits_np = np.asarray(bits_p, dtype=int)

        Ns_sym = L * Ns
        nsym = bits_np.size // L

        # Build symbol indices
        bit_mat = bits_np.reshape(nsym, L)
        weights = (1 << np.arange(L - 1, -1, -1)).astype(int)
        sym_index = (bit_mat * weights).sum(axis=1).astype(int)

        f0 = fc - ((M - 1) / 2.0) * (fd / Tb)
        freqs = (f0 + (fd / Tb) * np.arange(M, dtype=float)).tolist()
        f_sym = np.asarray(freqs, dtype=float)[sym_index]

        warnings = _warn_params(params, extra_freqs=[float(freqs[0]), float(freqs[-1])])

        N = nsym * Ns_sym
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        f_samp = np.repeat(f_sym, Ns_sym)
        s = Ac * np.cos(f_samp * omega)

        # meta fields expected/used by tests: L, M, sym_index, (and demod chosen_idx match)
        meta.update(
            {
                "L": L,
                "M": M,
                "fd": fd,
                "f0": float(f0),
                "freqs": freqs,
                "pad_bits": pad,
                "sym_bits": [tuple(map(int, row)) for row in bit_mat.tolist()],
                "sym_index": sym_index.astype(int).tolist(),
                "f_used": f_sym.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return s.astype(float, copy=False), meta

    if scheme_u == "BPSK":
        phase1 = float(kwargs.get("phase1", np.pi))
        phase0 = float(kwargs.get("phase0", 0.0))

        bits_np = np.asarray(bits, dtype=int)
        phases_bit = np.where(bits_np == 1, phase1, phase0).astype(float)

        N = bits_np.size * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        c = np.cos(fc * omega)
        sref = np.sin(fc * omega)

        cosphi = np.repeat(np.cos(phases_bit), Ns)
        sinphi = np.repeat(np.sin(phases_bit), Ns)

        sig = Ac * (c * cosphi - sref * sinphi)

        meta.update({"phase1": phase1, "phase0": phase0, "phases": phases_bit.tolist(), "warnings": warnings})
        return sig.astype(float, copy=False), meta

    if scheme_u == "DPSK":
        phase_init = float(kwargs.get("phase_init", 0.0))
        delta_phase = float(kwargs.get("delta_phase", np.pi))

        bits_np = np.asarray(bits, dtype=int)
        phases_bit = np.empty(bits_np.size, dtype=float)
        prev = phase_init
        for i, b in enumerate(bits_np.tolist()):
            if b == 1:
                prev = prev + delta_phase
            phases_bit[i] = prev

        N = bits_np.size * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        c = np.cos(fc * omega)
        sref = np.sin(fc * omega)

        cosphi = np.repeat(np.cos(phases_bit), Ns)
        sinphi = np.repeat(np.sin(phases_bit), Ns)
        sig = Ac * (c * cosphi - sref * sinphi)

        meta.update(
            {
                "phase_init": phase_init,
                "delta_phase": delta_phase,
                "phases": phases_bit.tolist(),
                "warnings": warnings,
            }
        )
        return sig.astype(float, copy=False), meta

    if scheme_u == "QPSK":
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        bits_p, pad = _pad_bits(bits, 2)
        bits_np = np.asarray(bits_p, dtype=int)

        pairs = bits_np.reshape(-1, 2)
        # mapping simplifies: I = 2*b0-1, Q=2*b1-1 (matches _QPSK_MAP)
        I = (2 * pairs[:, 0] - 1).astype(float)
        Q = (2 * pairs[:, 1] - 1).astype(float)

        Ns_sym = 2 * Ns
        nsym = pairs.shape[0]
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega + phi_ref

        c = np.cos(phase)
        sref = np.sin(phase)

        I_rep = np.repeat(I, Ns_sym)
        Q_rep = np.repeat(Q, Ns_sym)

        sig = (Ac / np.sqrt(2.0)) * (I_rep * c - Q_rep * sref)

        meta.update(
            {
                "phi_ref": phi_ref,
                "pad_bits": pad,
                "symbols": nsym,
                "sym_bits": [tuple(map(int, row)) for row in pairs.tolist()],
                "I": I.tolist(),
                "Q": Q.tolist(),
                "warnings": warnings,
            }
        )
        return sig.astype(float, copy=False), meta

    if scheme_u == "QAM":
        axis_levels = int(kwargs.get("axis_levels", 2))
        phi_ref = float(kwargs.get("phi_ref", 0.0))

        if axis_levels not in (2, 4):
            raise ValueError("QAM: axis_levels must be 2 or 4")

        bits_per_axis = 1 if axis_levels == 2 else 2
        bits_per_symbol = 2 * bits_per_axis

        bits_p, pad = _pad_bits(bits, bits_per_symbol)
        bits_np = np.asarray(bits_p, dtype=int)

        I_stream = bits_np[0::2]
        Q_stream = bits_np[1::2]

        nsym = I_stream.size // bits_per_axis
        I_bits = I_stream.reshape(nsym, bits_per_axis)
        Q_bits = Q_stream.reshape(nsym, bits_per_axis)

        if axis_levels == 2:
            I = (2 * I_bits[:, 0] - 1).astype(float)
            Q = (2 * Q_bits[:, 0] - 1).astype(float)
            norm = 1.0
        else:
            idxI = (2 * I_bits[:, 0] + I_bits[:, 1]).astype(int)
            idxQ = (2 * Q_bits[:, 0] + Q_bits[:, 1]).astype(int)
            table = np.array([-3.0, -1.0, +3.0, +1.0], dtype=float)
            I = table[idxI]
            Q = table[idxQ]
            norm = np.sqrt(10.0)

        Ns_sym = bits_per_symbol * Ns
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega + phi_ref
        c = np.cos(phase)
        sref = np.sin(phase)

        I_rep = np.repeat(I, Ns_sym)
        Q_rep = np.repeat(Q, Ns_sym)

        sig = Ac * ((I_rep / norm) * c + (Q_rep / norm) * sref)

        if bits_per_axis == 1:
            sym_bits_interleaved = [[int(I_bits[i, 0]), int(Q_bits[i, 0])] for i in range(nsym)]
        else:
            sym_bits_interleaved = [
                [int(I_bits[i, 0]), int(Q_bits[i, 0]), int(I_bits[i, 1]), int(Q_bits[i, 1])]
                for i in range(nsym)
            ]

        meta.update(
            {
                "axis_levels": axis_levels,
                "bits_per_axis": bits_per_axis,
                "bits_per_symbol": bits_per_symbol,
                "phi_ref": phi_ref,
                "pad_bits": pad,
                "symbols": nsym,
                "sym_bits": sym_bits_interleaved,
                "I": I.tolist(),
                "Q": Q.tolist(),
                "norm": float(norm),
                "warnings": warnings,
            }
        )
        return sig.astype(float, copy=False), meta

    if scheme_u == "16QAM":
        # Legacy scheme
        bits_p, pad = _pad_bits(bits, 4)
        bits_np = np.asarray(bits_p, dtype=int)
        quads = bits_np.reshape(-1, 4)

        bI = quads[:, 0:2]
        bQ = quads[:, 2:4]

        idxI = (2 * bI[:, 0] + bI[:, 1]).astype(int)
        idxQ = (2 * bQ[:, 0] + bQ[:, 1]).astype(int)
        table = np.array([-3.0, -1.0, +3.0, +1.0], dtype=float)
        I = table[idxI]
        Q = table[idxQ]

        Ns_sym = 4 * Ns
        nsym = quads.shape[0]
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega
        c = np.cos(phase)
        sref = np.sin(phase)

        I_rep = np.repeat(I, Ns_sym)
        Q_rep = np.repeat(Q, Ns_sym)
        norm = np.sqrt(10.0)

        sig = Ac * ((I_rep / norm) * c + (Q_rep / norm) * sref)

        meta.update(
            {
                "pad_bits": pad,
                "symbols": nsym,
                "sym_bits": [tuple(map(int, row)) for row in quads.tolist()],
                "I": I.tolist(),
                "Q": Q.tolist(),
                "norm": float(norm),
                "warnings": warnings,
            }
        )
        return sig.astype(float, copy=False), meta

    raise ValueError(f"Unknown modulation scheme: {scheme}")


# ===============================================================
# Core API: demodulate(s_t, scheme, params, **kwargs) -> (bits,meta)
# ===============================================================

def demodulate(s_t: np.ndarray, scheme: str, params: SimParams, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
    scheme_u = scheme.upper().strip()
    Ns = int(params.samples_per_bit)
    Tb = float(params.Tb)
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)

    meta: Dict[str, Any] = {"scheme": scheme_u}
    warnings: List[str] = _warn_params(params, extra_freqs=[])

    s_t = np.asarray(s_t, dtype=float)

    if scheme_u == "ASK":
        A0 = float(kwargs.get("A0", 0.5))
        A1 = float(kwargs.get("A1", 1.0))
        if A1 <= A0:
            warnings.append("ASK: A1 <= A0; thresholding may be ambiguous.")

        seg = _segment_view(s_t, Ns)
        nbits = seg.shape[0]

        N = nbits * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        c = np.cos(fc * omega).reshape(nbits, Ns)
        sref = np.sin(fc * omega).reshape(nbits, Ns)

        I = (2.0 / Ns) * np.sum(seg * c, axis=1)
        Q = (2.0 / Ns) * np.sum(seg * sref, axis=1)

        A_est = np.sqrt(I * I + Q * Q) / (Ac if Ac != 0 else 1.0)
        thr = 0.5 * (A0 + A1)
        bits_out = (A_est > thr).astype(int)

        meta.update(
            {
                "A0": A0,
                "A1": A1,
                "threshold": thr,
                "A_hat": A_est.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "BFSK":
        tone_sep = float(kwargs.get("tone_sep", 4.0))
        f0 = float(kwargs.get("f0", fc - tone_sep / Tb))
        f1 = float(kwargs.get("f1", fc + tone_sep / Tb))

        if f1 < f0:
            f0, f1 = f1, f0
        warnings = _warn_params(params, extra_freqs=[f0, f1])

        seg = _segment_view(s_t, Ns)
        nbits = seg.shape[0]
        N = nbits * Ns
        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        c0 = np.cos(f0 * omega).reshape(nbits, Ns)
        s0 = np.sin(f0 * omega).reshape(nbits, Ns)
        c1 = np.cos(f1 * omega).reshape(nbits, Ns)
        s1 = np.sin(f1 * omega).reshape(nbits, Ns)

        I0 = (2.0 / Ns) * np.sum(seg * c0, axis=1)
        Q0 = (2.0 / Ns) * np.sum(seg * s0, axis=1)
        I1 = (2.0 / Ns) * np.sum(seg * c1, axis=1)
        Q1 = (2.0 / Ns) * np.sum(seg * s1, axis=1)

        E0 = I0 * I0 + Q0 * Q0
        E1 = I1 * I1 + Q1 * Q1
        bits_out = (E1 > E0).astype(int)

        meta.update(
            {
                "tone_sep": tone_sep,
                "f0": f0,
                "f1": f1,
                "E0": E0.astype(float).tolist(),
                "E1": E1.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "MFSK":
        L = int(kwargs.get("L", 2))
        if L <= 0:
            raise ValueError("MFSK: L must be >= 1")
        M = 2**L
        fd = float(kwargs.get("fd", 2.0))

        Ns_sym = L * Ns
        seg = _segment_view(s_t, Ns_sym)
        nsym = seg.shape[0]
        N = nsym * Ns_sym

        f0 = fc - ((M - 1) / 2.0) * (fd / Tb)
        freqs = (f0 + (fd / Tb) * np.arange(M, dtype=float))
        warnings = _warn_params(params, extra_freqs=[float(freqs[0]), float(freqs[-1])])

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        energies = np.empty((nsym, M), dtype=float)
        for i, f in enumerate(freqs.tolist()):
            c = np.cos(f * omega).reshape(nsym, Ns_sym)
            sref = np.sin(f * omega).reshape(nsym, Ns_sym)
            I = (2.0 / Ns_sym) * np.sum(seg * c, axis=1)
            Q = (2.0 / Ns_sym) * np.sum(seg * sref, axis=1)
            energies[:, i] = I * I + Q * Q

        chosen_idx = np.argmax(energies, axis=1).astype(int)

        # Convert symbol indices back to bits (natural binary, MSB-first)
        out_bits = []
        for idx in chosen_idx.tolist():
            for sh in range(L - 1, -1, -1):
                out_bits.append((idx >> sh) & 1)

        meta.update(
            {
                "L": L,
                "M": M,
                "fd": fd,
                "f0": float(f0),
                "freqs": freqs.astype(float).tolist(),
                "energies": energies.astype(float).tolist(),
                "chosen_idx": chosen_idx.tolist(),
                "warnings": warnings,
            }
        )
        return out_bits, meta

    if scheme_u == "BPSK":
        phase1 = float(kwargs.get("phase1", np.pi))
        phase0 = float(kwargs.get("phase0", 0.0))

        seg = _segment_view(s_t, Ns)
        nbits = seg.shape[0]
        N = nbits * Ns

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        c = np.cos(fc * omega).reshape(nbits, Ns)
        sref = np.sin(fc * omega).reshape(nbits, Ns)

        I = (2.0 / Ns) * np.sum(seg * c, axis=1)
        Q = (2.0 / Ns) * np.sum(seg * sref, axis=1)

        I_norm = I / (Ac if Ac != 0 else 1.0)
        phi = np.arctan2(-Q, I)

        d1 = _ang_dist(phi, phase1)
        d0 = _ang_dist(phi, phase0)
        bits_out = (d1 < d0).astype(int)

        meta.update(
            {
                "phase1": phase1,
                "phase0": phase0,
                "I_hat": I_norm.astype(float).tolist(),
                "phi_hat": phi.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "DPSK":
        phase_init = float(kwargs.get("phase_init", 0.0))
        delta_phase = float(kwargs.get("delta_phase", np.pi))

        seg = _segment_view(s_t, Ns)
        nbits = seg.shape[0]
        N = nbits * Ns

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t

        c = np.cos(fc * omega).reshape(nbits, Ns)
        sref = np.sin(fc * omega).reshape(nbits, Ns)

        I = (2.0 / Ns) * np.sum(seg * c, axis=1)
        Q = (2.0 / Ns) * np.sum(seg * sref, axis=1)

        phi = np.arctan2(-Q, I)

        prev = np.empty_like(phi)
        prev[0] = phase_init
        if nbits > 1:
            prev[1:] = phi[:-1]
        dphi = _wrap_angle(phi - prev)

        target = abs(delta_phase)
        bits_out = (np.abs(np.abs(dphi) - target) < np.abs(dphi)).astype(int)

        meta.update(
            {
                "phase_init": phase_init,
                "delta_phase": delta_phase,
                "phi_hat": phi.astype(float).tolist(),
                "delta_hat": dphi.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "QPSK":
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        Ns_sym = 2 * Ns

        seg = _segment_view(s_t, Ns_sym)
        nsym = seg.shape[0]
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega + phi_ref

        c = np.cos(phase).reshape(nsym, Ns_sym)
        sref = np.sin(phase).reshape(nsym, Ns_sym)

        I = (2.0 / Ns_sym) * np.sum(seg * c, axis=1)
        Q = (2.0 / Ns_sym) * np.sum(seg * sref, axis=1)

        I_sym = (I * np.sqrt(2.0)) / (Ac if Ac != 0 else 1.0)
        Q_sym = (-Q * np.sqrt(2.0)) / (Ac if Ac != 0 else 1.0)

        b0 = (I_sym >= 0).astype(int)
        b1 = (Q_sym >= 0).astype(int)

        bits_out = np.empty(2 * nsym, dtype=int)
        bits_out[0::2] = b0
        bits_out[1::2] = b1

        meta.update(
            {
                "phi_ref": phi_ref,
                "symbols": int(nsym),
                "I_hat": I_sym.astype(float).tolist(),
                "Q_hat": Q_sym.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "QAM":
        axis_levels = int(kwargs.get("axis_levels", 2))
        phi_ref = float(kwargs.get("phi_ref", 0.0))

        if axis_levels not in (2, 4):
            raise ValueError("QAM: axis_levels must be 2 or 4")

        bits_per_axis = 1 if axis_levels == 2 else 2
        bits_per_symbol = 2 * bits_per_axis
        Ns_sym = bits_per_symbol * Ns

        seg = _segment_view(s_t, Ns_sym)
        nsym = seg.shape[0]
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega + phi_ref
        c = np.cos(phase).reshape(nsym, Ns_sym)
        sref = np.sin(phase).reshape(nsym, Ns_sym)

        Icorr = (2.0 / Ns_sym) * np.sum(seg * c, axis=1)
        Qcorr = (2.0 / Ns_sym) * np.sum(seg * sref, axis=1)

        if axis_levels == 2:
            norm = 1.0
            I_hat = Icorr / (Ac if Ac != 0 else 1.0)
            Q_hat = Qcorr / (Ac if Ac != 0 else 1.0)

            I_dec = np.where(I_hat >= 0.0, +1.0, -1.0)
            Q_dec = np.where(Q_hat >= 0.0, +1.0, -1.0)

            bI = (I_dec > 0).astype(int)
            bQ = (Q_dec > 0).astype(int)

            bits_out = np.empty(2 * nsym, dtype=int)
            bits_out[0::2] = bI
            bits_out[1::2] = bQ
        else:
            norm = np.sqrt(10.0)
            I_hat = (Icorr * norm) / (Ac if Ac != 0 else 1.0)
            Q_hat = (Qcorr * norm) / (Ac if Ac != 0 else 1.0)

            I_dec = _nearest_16qam_level_vec(I_hat)
            Q_dec = _nearest_16qam_level_vec(Q_hat)

            # Map decision levels -> idx -> bits
            idxI = np.where(I_dec == -3.0, 0, np.where(I_dec == -1.0, 1, np.where(I_dec == 3.0, 2, 3))).astype(int)
            idxQ = np.where(Q_dec == -3.0, 0, np.where(Q_dec == -1.0, 1, np.where(Q_dec == 3.0, 2, 3))).astype(int)

            bI0 = (idxI // 2).astype(int)
            bI1 = (idxI % 2).astype(int)
            bQ0 = (idxQ // 2).astype(int)
            bQ1 = (idxQ % 2).astype(int)

            bits_out = np.empty(4 * nsym, dtype=int)
            # Original QAM returns in interleaved order: [I0, Q0, I1, Q1] per symbol
            bits_out[0::4] = bI0
            bits_out[1::4] = bQ0
            bits_out[2::4] = bI1
            bits_out[3::4] = bQ1

        meta.update(
            {
                "axis_levels": axis_levels,
                "bits_per_axis": bits_per_axis,
                "bits_per_symbol": bits_per_symbol,
                "phi_ref": phi_ref,
                "symbols": int(nsym),
                "norm": float(norm),
                "I_hat": I_hat.astype(float).tolist(),
                "Q_hat": Q_hat.astype(float).tolist(),
                "I_dec": I_dec.astype(float).tolist(),
                "Q_dec": Q_dec.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    if scheme_u == "16QAM":
        # Legacy demod: no phi_ref
        Ns_sym = 4 * Ns
        seg = _segment_view(s_t, Ns_sym)
        nsym = seg.shape[0]
        N = nsym * Ns_sym

        t = make_time_axis(N, fs)
        omega = 2.0 * np.pi * t
        phase = fc * omega
        c = np.cos(phase).reshape(nsym, Ns_sym)
        sref = np.sin(phase).reshape(nsym, Ns_sym)

        Icorr = (2.0 / Ns_sym) * np.sum(seg * c, axis=1)
        Qcorr = (2.0 / Ns_sym) * np.sum(seg * sref, axis=1)

        norm = np.sqrt(10.0)
        I_hat = (Icorr * norm) / (Ac if Ac != 0 else 1.0)
        Q_hat = (Qcorr * norm) / (Ac if Ac != 0 else 1.0)

        I_dec = _nearest_16qam_level_vec(I_hat)
        Q_dec = _nearest_16qam_level_vec(Q_hat)

        idxI = np.where(I_dec == -3.0, 0, np.where(I_dec == -1.0, 1, np.where(I_dec == 3.0, 2, 3))).astype(int)
        idxQ = np.where(Q_dec == -3.0, 0, np.where(Q_dec == -1.0, 1, np.where(Q_dec == 3.0, 2, 3))).astype(int)

        bI0 = (idxI // 2).astype(int)
        bI1 = (idxI % 2).astype(int)
        bQ0 = (idxQ // 2).astype(int)
        bQ1 = (idxQ % 2).astype(int)

        bits_out = np.empty(4 * nsym, dtype=int)
        # 16QAM returns [I0, I1, Q0, Q1] per symbol (legacy layout)
        bits_out[0::4] = bI0
        bits_out[1::4] = bI1
        bits_out[2::4] = bQ0
        bits_out[3::4] = bQ1

        meta.update(
            {
                "symbols": int(nsym),
                "norm": float(norm),
                "I_hat": I_hat.astype(float).tolist(),
                "Q_hat": Q_hat.astype(float).tolist(),
                "I_dec": I_dec.astype(float).tolist(),
                "Q_dec": Q_dec.astype(float).tolist(),
                "warnings": warnings,
            }
        )
        return bits_out.tolist(), meta

    raise ValueError(f"Unknown demodulation scheme: {scheme}")


# =================================================
# Public wrapper: simulate_d2a(...) -> SimResult
# =================================================

def simulate_d2a(bits: List[int], scheme: str, params: SimParams, **kwargs) -> SimResult:
    """
    Matches the original `simulate_d2a` contract in comm_sim/d2a.py:
    - Modulate -> Demodulate
    - Trim decoded bits back to original input length
    - Return SimResult with signals/bits/meta dicts
    """
    _validate_bits(bits)

    tx, meta_tx = modulate(bits, scheme, params, **kwargs)
    rx_bits, meta_rx = demodulate(tx, scheme, params, **kwargs)

    rx_trim = rx_bits[: len(bits)]
    t = make_time_axis(len(tx), params.fs)

    warnings = list(meta_tx.get("warnings", [])) + list(meta_rx.get("warnings", []))
    meta = {
        "scheme": scheme.upper().strip(),
        "modulate": meta_tx,
        "demodulate": meta_rx,
        "input_len": len(bits),
        "decoded_len": len(rx_trim),
        "pad_bits": int(meta_tx.get("pad_bits", 0)),
        "match": (rx_trim == list(bits)),
        "warnings": warnings,
    }

    return SimResult(
        t=t,
        signals={"tx": tx},
        bits={"input": list(bits), "decoded": rx_trim},
        meta=meta,
    )
