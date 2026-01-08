from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from utils import SimParams, SimResult, make_time_axis

# ----------------------------
# Small utilities / validation
# ----------------------------

def _validate_bits(bits: List[int]) -> None:
    if not bits:
        raise ValueError("Bits list is empty.")
    if any(b not in (0, 1) for b in bits):
        raise ValueError("Bits must be a list of 0/1 integers.")


def _pad_bits(bits: List[int], k: int) -> Tuple[List[int], int]:
    """Pad with zeros to a multiple of k. Return (padded_bits, pad_count)."""
    pad = (-len(bits)) % k
    if pad:
        return bits + [0] * pad, pad
    return bits, 0

def _iq_correlator(seg: np.ndarray, tseg: np.ndarray, f: float) -> Tuple[float, float]:
    """
    Coherent I/Q correlator at frequency f.
    Returns (I, Q) such that for seg = A*cos(2π f t + φ),
    I ≈ A*cosφ, Q ≈ A*sinφ (assuming enough samples).
    """
    n = len(seg)
    if n == 0:
        return 0.0, 0.0
    c = np.cos(2 * np.pi * f * tseg)
    s = np.sin(2 * np.pi * f * tseg)
    # Scale by 2/n to undo the average cos^2 factor (~1/2)
    I = (2.0 / n) * float(np.dot(seg, c))
    Q = (2.0 / n) * float(np.dot(seg, s))
    return I, Q


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


# ----------------------------
# Mapping tables (exact inverse)
# ----------------------------

# QPSK Gray mapping (00,01,11,10)
# bits -> (I, Q)
_QPSK_MAP: Dict[Tuple[int, int], Tuple[float, float]] = {
    (0, 0): (+1.0, +1.0),
    (0, 1): (-1.0, +1.0),
    (1, 1): (-1.0, -1.0),
    (1, 0): (+1.0, -1.0),
}
# inverse: sign(I), sign(Q) -> bits
_QPSK_INV: Dict[Tuple[int, int], Tuple[int, int]] = {
    (+1, +1): (0, 0),
    (-1, +1): (0, 1),
    (-1, -1): (1, 1),
    (+1, -1): (1, 0),
}

# 16-QAM Gray per axis: 2 bits -> level in {-3,-1,+1,+3}
# 00->-3, 01->-1, 11->+1, 10->+3
_16QAM_AXIS_MAP: Dict[Tuple[int, int], float] = {
    (0, 0): -3.0,
    (0, 1): -1.0,
    (1, 1): +1.0,
    (1, 0): +3.0,
}
_16QAM_AXIS_INV: Dict[float, Tuple[int, int]] = {
    -3.0: (0, 0),
    -1.0: (0, 1),
    +1.0: (1, 1),
    +3.0: (1, 0),
}


def _sign01(x: float) -> int:
    """Return +1 if x>=0 else -1."""
    return +1 if x >= 0 else -1


def _nearest_16qam_level(x: float) -> float:
    """Nearest of {-3,-1,+1,+3}."""
    # thresholds at -2, 0, +2
    if x < -2.0:
        return -3.0
    if x < 0.0:
        return -1.0
    if x < 2.0:
        return +1.0
    return +3.0


# ----------------------------
# Modulation / Demodulation
# ----------------------------

def modulate(bits: List[int], scheme: str, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    _validate_bits(bits)

    scheme = scheme.upper()
    Ns = int(params.samples_per_bit)
    Tb = float(params.Tb)
    fs = float(params.fs)
    Ac = float(params.Ac)
    fc = float(params.fc)

    # time axis per entire waveform (keeps continuous phase)
    N = len(bits) * Ns
    # For QPSK/16QAM, we'll resize after padding
    t = make_time_axis(N, fs)

    meta: Dict[str, Any] = {"scheme": scheme}
    warnings: List[str] = []

    if scheme == "ASK":
        A0 = float(kwargs.get("A0", 0.2))
        A1 = float(kwargs.get("A1", 1.0))
        if A1 <= A0:
            warnings.append("ASK: A1 <= A0; thresholding may be ambiguous.")
        warnings += _warn_params(params, extra_freqs=[])

        s = np.zeros(N, dtype=float)
        amps: List[float] = []
        for i, b in enumerate(bits):
            a, z = i * Ns, (i + 1) * Ns
            A = A1 if b == 1 else A0
            amps.append(A)
            seg_t = t[a:z]
            s[a:z] = (Ac * A) * np.cos(2 * np.pi * fc * seg_t)

        meta.update({"A0": A0, "A1": A1, "A_used": amps, "warnings": warnings})
        return s, meta

    if scheme == "BFSK":
        # Preferred: explicit f0/f1 (Hz) from UI
        if "f0" in kwargs and "f1" in kwargs:
            f0 = float(kwargs["f0"])
            f1 = float(kwargs["f1"])
            tone_sep = None
        else:
            # Backward compatible: tone_sep in (1/Tb units) as before
            tone_sep = float(kwargs.get("tone_sep", 2.0))
            f0 = fc - tone_sep / Tb
            f1 = fc + tone_sep / Tb

        # Basic sanity: avoid swapped frequencies
        if f1 < f0:
            f0, f1 = f1, f0

        warnings += _warn_params(params, extra_freqs=[f0, f1])

        s = np.zeros(N, dtype=float)
        freqs: List[float] = []
        for i, b in enumerate(bits):
            a, z = i * Ns, (i + 1) * Ns
            f = f1 if b == 1 else f0
            freqs.append(f)
            seg_t = t[a:z]
            s[a:z] = Ac * np.cos(2 * np.pi * f * seg_t)

        meta.update({
            "tone_sep": tone_sep,
            "f0": f0,
            "f1": f1,
            "f_used": freqs,
            "warnings": warnings
        })
        return s, meta
    
    if scheme == "MFSK":
        # MFSK parameters (Eq 5.4):
        #   M = 2^L, symbol holds L bits, Ts = L*Tb
        #   f_i = fc + (2i - 1 - M) * fd,  1 <= i <= M
        L = int(kwargs.get("L", 2))          # bits per symbol
        fd = float(kwargs.get("fd", 1.0))    # frequency difference (Hz)
        if L < 1:
            raise ValueError("MFSK: L must be >= 1.")
        M = 2 ** L

        bitsL, pad = _pad_bits(bits, L)

        Ns_sym = L * Ns
        nsym = len(bitsL) // L
        Nsym = nsym * Ns_sym
        tS = make_time_axis(Nsym, fs)

        # Precompute tone set
        freqs = [fc + (2 * (i + 1) - 1 - M) * fd for i in range(M)]
        warnings += _warn_params(params, extra_freqs=freqs)

        s = np.zeros(Nsym, dtype=float)
        sym_bits: List[List[int]] = []
        sym_index: List[int] = []
        f_used: List[float] = []

        for k in range(nsym):
            chunk = bitsL[k * L:(k + 1) * L]
            sym_bits.append(chunk)

            # Map bits to index i in [0..M-1] (natural binary)
            idx = 0
            for b in chunk:
                idx = (idx << 1) | int(b)
            sym_index.append(idx)

            f = float(freqs[idx])
            f_used.append(f)

            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg_t = tS[a:z]
            s[a:z] = Ac * np.cos(2 * np.pi * f * seg_t)

        meta.update({
            "L": L,
            "M": M,
            "fd": fd,
            "pad_bits": pad,
            "freqs": freqs,
            "sym_bits": sym_bits,
            "sym_index": sym_index,
            "f_used": f_used,
            "warnings": warnings
        })
        return s, meta

    if scheme == "BPSK":
        warnings += _warn_params(params, extra_freqs=[])

        # Phases (rad). Defaults implement textbook BPSK: bit1 at 0, bit0 at pi.
        phase1 = float(kwargs.get("phase1", 0.0))
        phase0 = float(kwargs.get("phase0", np.pi))

        s = np.zeros(N, dtype=float)
        phases: List[float] = []
        for i, b in enumerate(bits):
            a, z = i * Ns, (i + 1) * Ns
            phase = phase1 if b == 1 else phase0
            phases.append(phase)
            seg_t = t[a:z]
            s[a:z] = Ac * np.cos(2 * np.pi * fc * seg_t + phase)

        meta.update({"phase1": phase1, "phase0": phase0, "phase_used": phases, "warnings": warnings})
        return s, meta

    if scheme == "QPSK":
        bits2, pad = _pad_bits(bits, 2)
        N2 = len(bits2) * Ns
        t2 = make_time_axis(N2, fs)
        s = np.zeros(N2, dtype=float)

        Ns_sym = 2 * Ns
        nsym = len(bits2) // 2

        I_levels: List[float] = []
        Q_levels: List[float] = []
        sym_bits: List[Tuple[int, int]] = []
        warnings += _warn_params(params, extra_freqs=[])

        for k in range(nsym):
            b0, b1 = bits2[2 * k], bits2[2 * k + 1]
            I, Q = _QPSK_MAP[(b0, b1)]
            sym_bits.append((b0, b1))
            I_levels.append(I)
            Q_levels.append(Q)

            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg_t = t2[a:z]
            c = np.cos(2 * np.pi * fc * seg_t)
            sn = np.sin(2 * np.pi * fc * seg_t)
            s[a:z] = Ac * (I * c + Q * sn)

        meta.update({
            "pad_bits": pad,
            "symbols": nsym,
            "sym_bits": sym_bits,
            "I": I_levels,
            "Q": Q_levels,
            "warnings": warnings,
        })
        return s, meta

    if scheme == "16QAM":
        bits4, pad = _pad_bits(bits, 4)
        N4 = len(bits4) * Ns
        t4 = make_time_axis(N4, fs)
        s = np.zeros(N4, dtype=float)

        Ns_sym = 4 * Ns
        nsym = len(bits4) // 4
        # Normalize by 3 so axis levels stay within [-1,+1] multipliers
        norm = 3.0
        warnings += _warn_params(params, extra_freqs=[])

        I_levels: List[float] = []
        Q_levels: List[float] = []
        sym_bits: List[Tuple[int, int, int, int]] = []

        for k in range(nsym):
            quad = bits4[4 * k: 4 * k + 4]
            bI = (quad[0], quad[1])
            bQ = (quad[2], quad[3])
            I = _16QAM_AXIS_MAP[bI]
            Q = _16QAM_AXIS_MAP[bQ]
            sym_bits.append((quad[0], quad[1], quad[2], quad[3]))
            I_levels.append(I)
            Q_levels.append(Q)

            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg_t = t4[a:z]
            c = np.cos(2 * np.pi * fc * seg_t)
            sn = np.sin(2 * np.pi * fc * seg_t)
            s[a:z] = Ac * ((I / norm) * c + (Q / norm) * sn)

        meta.update({
            "pad_bits": pad,
            "symbols": nsym,
            "sym_bits": sym_bits,
            "I": I_levels,
            "Q": Q_levels,
            "norm": norm,
            "warnings": warnings,
        })
        return s, meta

    raise ValueError(f"Unknown modulation scheme: {scheme}")


def demodulate(s_t: np.ndarray, scheme: str, params: SimParams, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
    scheme = scheme.upper()
    Ns = int(params.samples_per_bit)
    Tb = float(params.Tb)
    fs = float(params.fs)
    Ac = float(params.Ac)
    fc = float(params.fc)

    meta: Dict[str, Any] = {"scheme": scheme}
    warnings: List[str] = []

    N = len(s_t)
    t = make_time_axis(N, fs)

    if scheme == "ASK":
        A0 = float(kwargs.get("A0", 0.2))
        A1 = float(kwargs.get("A1", 1.0))
        thr = 0.5 * (A0 + A1)
        warnings += _warn_params(params, extra_freqs=[])

        nbits = N // Ns
        bits_out: List[int] = []
        A_hat: List[float] = []
        for i in range(nbits):
            a, z = i * Ns, (i + 1) * Ns
            seg = s_t[a:z]
            seg_t = t[a:z]
            I, Q = _iq_correlator(seg, seg_t, fc)
            A_est = np.sqrt(I * I + Q * Q) / (Ac if Ac != 0 else 1.0)
            A_hat.append(float(A_est))
            bits_out.append(1 if A_est >= thr else 0)

        meta.update({"A0": A0, "A1": A1, "thr": thr, "A_hat": A_hat, "warnings": warnings})
        return bits_out, meta

    if scheme == "BFSK":
        if "f0" in kwargs and "f1" in kwargs:
            f0 = float(kwargs["f0"])
            f1 = float(kwargs["f1"])
            tone_sep = None
        else:
            tone_sep = float(kwargs.get("tone_sep", 2.0))
            f0 = fc - tone_sep / Tb
            f1 = fc + tone_sep / Tb

        if f1 < f0:
            f0, f1 = f1, f0

        warnings += _warn_params(params, extra_freqs=[f0, f1])

        nbits = N // Ns
        bits_out: List[int] = []
        E0_list: List[float] = []
        E1_list: List[float] = []

        for i in range(nbits):
            a, z = i * Ns, (i + 1) * Ns
            seg = s_t[a:z]
            seg_t = t[a:z]

            I0, Q0 = _iq_correlator(seg, seg_t, f0)
            I1, Q1 = _iq_correlator(seg, seg_t, f1)
            E0 = I0 * I0 + Q0 * Q0
            E1 = I1 * I1 + Q1 * Q1
            E0_list.append(float(E0))
            E1_list.append(float(E1))
            bits_out.append(1 if E1 > E0 else 0)

        meta.update({
            "tone_sep": tone_sep,
            "f0": f0,
            "f1": f1,
            "E0": E0_list,
            "E1": E1_list,
            "warnings": warnings
        })
        return bits_out, meta

    if scheme == "MFSK":
        L = int(kwargs.get("L", 2))
        fd = float(kwargs.get("fd", 1.0))
        if L < 1:
            raise ValueError("MFSK: L must be >= 1.")
        M = 2 ** L

        Ns_sym = L * Ns
        nsym = N // Ns_sym

        # Tone set
        freqs = [fc + (2 * (i + 1) - 1 - M) * fd for i in range(M)]
        warnings += _warn_params(params, extra_freqs=freqs)

        bits_out: List[int] = []
        chosen_idx: List[int] = []
        E_list: List[List[float]] = []

        for k in range(nsym):
            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg = s_t[a:z]
            seg_t = t[a:z]

            energies: List[float] = []
            for f in freqs:
                I, Q = _iq_correlator(seg, seg_t, float(f))
                energies.append(float(I * I + Q * Q))

            idx = int(np.argmax(energies))
            chosen_idx.append(idx)
            E_list.append(energies)

            # idx -> L bits (natural binary)
            for shift in range(L - 1, -1, -1):
                bits_out.append((idx >> shift) & 1)

        meta.update({
            "L": L,
            "M": M,
            "fd": fd,
            "freqs": freqs,
            "chosen_idx": chosen_idx,
            "energies": E_list,
            "warnings": warnings
        })
        return bits_out, meta

    if scheme == "BPSK":
        warnings += _warn_params(params, extra_freqs=[])

        # Must match TX
        phase1 = float(kwargs.get("phase1", 0.0))
        phase0 = float(kwargs.get("phase0", np.pi))

        def _ang_dist(a: float, b: float) -> float:
            # shortest angular distance on circle
            d = (a - b + np.pi) % (2 * np.pi) - np.pi
            return abs(d)

        nbits = N // Ns
        bits_out: List[int] = []
        I_hat: List[float] = []

        for i in range(nbits):
            a, z = i * Ns, (i + 1) * Ns
            seg = s_t[a:z]
            seg_t = t[a:z]

            I, Q = _iq_correlator(seg, seg_t, fc)

            # Normalize I for display/debug (kept from your original code)
            I_norm = I / (Ac if Ac != 0 else 1.0)
            I_hat.append(float(I_norm))

            # Decide by phase: compare to phase=0 (bit 1) vs phase=phase0 (bit 0)
            phi = float(np.arctan2(-Q, I))  # correct phase for cos(ωt+φ) with this IQ convention
            d1 = _ang_dist(phi, phase1)
            d0 = _ang_dist(phi, phase0)
            bits_out.append(1 if d1 <= d0 else 0)

        meta.update({"phase1": phase1, "phase0": phase0, "I_hat": I_hat, "warnings": warnings})
        return bits_out, meta

    if scheme == "QPSK":
        Ns_sym = 2 * Ns
        nsym = N // Ns_sym
        warnings += _warn_params(params, extra_freqs=[])

        bits_out: List[int] = []
        I_hat: List[float] = []
        Q_hat: List[float] = []

        for k in range(nsym):
            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg = s_t[a:z]
            seg_t = t[a:z]

            I, Q = _iq_correlator(seg, seg_t, fc)
            I_norm = I / (Ac if Ac != 0 else 1.0)
            Q_norm = Q / (Ac if Ac != 0 else 1.0)
            I_hat.append(float(I_norm))
            Q_hat.append(float(Q_norm))

            si = _sign01(I_norm)
            sq = _sign01(Q_norm)
            b0, b1 = _QPSK_INV[(si, sq)]
            bits_out.extend([b0, b1])

        meta.update({"symbols": nsym, "I_hat": I_hat, "Q_hat": Q_hat, "warnings": warnings})
        return bits_out, meta

    if scheme == "16QAM":
        Ns_sym = 4 * Ns
        nsym = N // Ns_sym
        norm = 3.0
        warnings += _warn_params(params, extra_freqs=[])

        bits_out: List[int] = []
        I_hat: List[float] = []
        Q_hat: List[float] = []
        I_dec: List[float] = []
        Q_dec: List[float] = []

        for k in range(nsym):
            a, z = k * Ns_sym, (k + 1) * Ns_sym
            seg = s_t[a:z]
            seg_t = t[a:z]

            I, Q = _iq_correlator(seg, seg_t, fc)
            # Undo scaling: I ≈ Ac*(I_level/norm), so I_level ≈ (I * norm / Ac)
            I_level = (I * norm) / (Ac if Ac != 0 else 1.0)
            Q_level = (Q * norm) / (Ac if Ac != 0 else 1.0)
            I_hat.append(float(I_level))
            Q_hat.append(float(Q_level))

            I_lv = _nearest_16qam_level(I_level)
            Q_lv = _nearest_16qam_level(Q_level)
            I_dec.append(float(I_lv))
            Q_dec.append(float(Q_lv))

            bI = _16QAM_AXIS_INV[I_lv]
            bQ = _16QAM_AXIS_INV[Q_lv]
            bits_out.extend([bI[0], bI[1], bQ[0], bQ[1]])

        meta.update({
            "symbols": nsym,
            "I_hat": I_hat, "Q_hat": Q_hat,
            "I_dec": I_dec, "Q_dec": Q_dec,
            "norm": norm,
            "warnings": warnings
        })
        return bits_out, meta

    raise ValueError(f"Unknown demodulation scheme: {scheme}")


# ----------------------------
# End-to-end simulation wrapper
# ----------------------------

def simulate_d2a(bits: List[int], scheme: str, params: SimParams, **kwargs) -> SimResult:
    _validate_bits(bits)

    tx, meta_tx = modulate(bits, scheme, params, **kwargs)
    rx_bits, meta_rx = demodulate(tx, scheme, params, **kwargs)

    # If modulation pads bits (QPSK/16QAM), trim decode to original length
    rx_trim = rx_bits[:len(bits)]

    t = make_time_axis(len(tx), params.fs)

    meta: Dict[str, Any] = {
        "scheme": scheme.upper(),
        "modulate": meta_tx,
        "demodulate": meta_rx,
        "input_len": len(bits),
        "decoded_len": len(rx_trim),
        "pad_bits": int(meta_tx.get("pad_bits", 0)),
        "match": (rx_trim == bits),
        "warnings": (meta_tx.get("warnings", []) + meta_rx.get("warnings", [])),
    }

    return SimResult(
        t=t,
        signals={"tx": tx},
        bits={"input": bits, "decoded": rx_trim},
        meta=meta,
    )
