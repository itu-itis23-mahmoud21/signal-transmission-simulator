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
# Vectorized Helpers
# ----------------------------

def _vec_iq_correlator(s_t: np.ndarray, t: np.ndarray, f: float, Ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized I/Q correlator.
    Reshapes s_t into (num_sym, Ns) and correlates with cos/sin(2pi*f*t).
    Returns arrays of I and Q values (one per symbol).
    """
    n = len(s_t)
    # Ensure divisible
    if n % Ns != 0:
         # Should not happen in controlled sim, but handle safety
         valid = (n // Ns) * Ns
         s_t = s_t[:valid]
         t = t[:valid]
         n = valid

    num_sym = n // Ns
    
    # Generate full carrier references
    arg = 2 * np.pi * f * t
    c = np.cos(arg)
    s_ref = np.sin(arg)
    
    # Multiply
    prod_c = s_t * c
    prod_s = s_t * s_ref
    
    # Reshape and integrate (sum)
    # Shape: (num_sym, Ns)
    I_integrals = prod_c.reshape(num_sym, Ns).sum(axis=1)
    Q_integrals = prod_s.reshape(num_sym, Ns).sum(axis=1)
    
    # Scale: (2/N) * sum
    factor = 2.0 / Ns
    return I_integrals * factor, Q_integrals * factor


def _nearest_level_vec(vals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Find nearest level for each value in vals."""
    # broadcast abs diff: (N, 1) - (1, L) -> (N, L)
    diffs = np.abs(vals[:, None] - levels[None, :])
    idx = np.argmin(diffs, axis=1)
    return levels[idx]


# ----------------------------
# Modulation (Vectorized)
# ----------------------------

def modulate(bits: List[int], scheme: str, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    _validate_bits(bits)
    scheme = scheme.upper()
    Ns = int(params.samples_per_bit)
    fs = float(params.fs)
    Ac = float(params.Ac)
    fc = float(params.fc)

    meta: Dict[str, Any] = {"scheme": scheme}
    warnings: List[str] = []

    # Convert to numpy for vector ops
    bits_arr = np.array(bits, dtype=int)
    N_bits = len(bits_arr)

    if scheme == "ASK":
        A0 = float(kwargs.get("A0", 0.2))
        A1 = float(kwargs.get("A1", 1.0))
        if A1 <= A0:
            warnings.append("ASK: A1 <= A0; thresholding may be ambiguous.")
        warnings += _warn_params(params, extra_freqs=[])

        # Map bits to Amplitudes
        amps = np.where(bits_arr == 1, A1, A0)
        
        # Expand to samples: repeat each amplitude Ns times
        amps_full = np.repeat(amps, Ns)
        
        t = make_time_axis(len(amps_full), fs)
        s = (Ac * amps_full) * np.cos(2 * np.pi * fc * t)

        meta.update({"A0": A0, "A1": A1, "A_used": amps.tolist(), "warnings": warnings})
        return s, meta

    if scheme == "BFSK":
        if "f0" in kwargs and "f1" in kwargs:
            f0 = float(kwargs["f0"])
            f1 = float(kwargs["f1"])
            tone_sep = None
            # Ensure canonical order f0 < f1 per test expectation
            if f1 < f0:
                f0, f1 = f1, f0
        else:
            Tb = float(params.Tb)
            tone_sep = float(kwargs.get("tone_sep", 2.0))
            f0 = fc - tone_sep / Tb
            f1 = fc + tone_sep / Tb
            if f1 < f0: f0, f1 = f1, f0

        warnings += _warn_params(params, extra_freqs=[f0, f1])

        # Map bits to Frequencies
        freqs = np.where(bits_arr == 1, f1, f0)
        
        # Expand
        freqs_full = np.repeat(freqs, Ns)
        
        t = make_time_axis(len(freqs_full), fs)
        s = Ac * np.cos(2 * np.pi * freqs_full * t)

        meta.update({
            "tone_sep": tone_sep, "f0": f0, "f1": f1,
            "f_used": freqs.tolist(), "warnings": warnings
        })
        return s, meta

    if scheme == "MFSK":
        L = int(kwargs.get("L", 2))
        fd = float(kwargs.get("fd", 1.0))
        if L < 1: raise ValueError("MFSK: L must be >= 1.")
        M = 2 ** L
        
        # Pad bits
        bits_list, pad = _pad_bits(bits, L)
        bits_arr = np.array(bits_list, dtype=int)
        
        # Reshape to (num_symbols, L)
        num_sym = len(bits_arr) // L
        sym_rows = bits_arr.reshape(num_sym, L)
        
        # Convert binary rows to integer indices
        powers = 2 ** np.arange(L - 1, -1, -1)
        sym_indices = sym_rows.dot(powers)
        
        # Frequencies
        freq_lut = fc + (2 * (np.arange(M) + 1) - 1 - M) * fd
        
        # Map indices to frequencies
        chosen_freqs = freq_lut[sym_indices]
        warnings += _warn_params(params, extra_freqs=freq_lut.tolist())
        
        Ns_sym = L * Ns
        freqs_full = np.repeat(chosen_freqs, Ns_sym)
        
        t = make_time_axis(len(freqs_full), fs)
        s = Ac * np.cos(2 * np.pi * freqs_full * t)

        meta.update({
            "L": L, "M": M, "fd": fd, "pad_bits": pad,
            "freqs": freq_lut.tolist(),
            "sym_bits": sym_rows.tolist(),
            "sym_index": sym_indices.tolist(),
            "f_used": chosen_freqs.tolist(),
            "warnings": warnings
        })
        return s, meta

    if scheme == "BPSK":
        warnings += _warn_params(params, extra_freqs=[])
        phase1 = float(kwargs.get("phase1", 0.0))
        phase0 = float(kwargs.get("phase0", np.pi))

        phases = np.where(bits_arr == 1, phase1, phase0)
        phases_full = np.repeat(phases, Ns)
        
        t = make_time_axis(len(phases_full), fs)
        s = Ac * np.cos(2 * np.pi * fc * t + phases_full)

        meta.update({"phase1": phase1, "phase0": phase0, "phase_used": phases.tolist(), "warnings": warnings})
        return s, meta

    if scheme == "DPSK":
        warnings += _warn_params(params, extra_freqs=[])
        phase_init = float(kwargs.get("phase_init", 0.0))
        delta_phase = float(kwargs.get("delta_phase", np.pi))

        # Differential encoding
        phase_changes = np.where(bits_arr == 1, delta_phase, 0.0)
        phases = np.cumsum(phase_changes) + phase_init
        
        phases_full = np.repeat(phases, Ns)
        t = make_time_axis(len(phases_full), fs)
        s = Ac * np.cos(2 * np.pi * fc * t + phases_full)

        meta.update({
            "phase_init": phase_init, "delta_phase": delta_phase,
            "phase_used": phases.tolist(), "warnings": warnings
        })
        return s, meta

    if scheme == "QPSK":
        bits_list, pad = _pad_bits(bits, 2)
        bits_arr = np.array(bits_list, dtype=int)
        
        num_sym = len(bits_arr) // 2
        sym_rows = bits_arr.reshape(num_sym, 2)
        
        b0 = sym_rows[:, 0]
        b1 = sym_rows[:, 1]
        I_vals = np.where(b0 == 1, 1.0, -1.0)
        Q_vals = np.where(b1 == 1, 1.0, -1.0)
        
        Ns_sym = 2 * Ns
        I_full = np.repeat(I_vals, Ns_sym)
        Q_full = np.repeat(Q_vals, Ns_sym)
        
        t = make_time_axis(len(I_full), fs)
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        warnings += _warn_params(params, extra_freqs=[])

        arg = 2 * np.pi * fc * t + phi_ref
        c = np.cos(arg)
        sn = np.sin(arg)
        
        s = (Ac / np.sqrt(2.0)) * (I_full * c - Q_full * sn)

        sym_bits_list = [tuple(r) for r in sym_rows.tolist()]
        
        meta.update({
            "phi_ref": phi_ref, "pad_bits": pad, "symbols": num_sym,
            "sym_bits": sym_bits_list,
            "I": I_vals.tolist(), "Q": Q_vals.tolist(), "warnings": warnings,
        })
        return s, meta

    if scheme == "QAM":
        axis_levels = int(kwargs.get("axis_levels", 2))
        
        # --- FIX: explicit validation to pass tests ---
        if axis_levels not in (2, 4):
            raise ValueError(f"QAM axis_levels must be 2 or 4, got {axis_levels}")
        # ---------------------------------------------
        
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        warnings += _warn_params(params, extra_freqs=[])

        bits_per_axis = 1 if axis_levels == 2 else 2
        bits_per_symbol = 2 * bits_per_axis
        
        bits_list, pad = _pad_bits(bits, bits_per_symbol)
        bits_arr = np.array(bits_list, dtype=int)
        
        I_stream = bits_arr[0::2]
        Q_stream = bits_arr[1::2]
        
        num_sym = len(I_stream) // bits_per_axis
        
        I_bits_mat = I_stream.reshape(num_sym, bits_per_axis)
        Q_bits_mat = Q_stream.reshape(num_sym, bits_per_axis)
        
        if axis_levels == 2:
            # 1 bit per axis. 0->-1, 1->1
            I_vals = np.where(I_bits_mat[:, 0] == 1, 1.0, -1.0)
            Q_vals = np.where(Q_bits_mat[:, 0] == 1, 1.0, -1.0)
            norm = 1.0
        else:
            # 2 bits per axis (16-QAM). 
            powers = np.array([2, 1])
            I_idx = I_bits_mat.dot(powers)
            Q_idx = Q_bits_mat.dot(powers)
            
            lut = np.array([-3.0, -1.0, 3.0, 1.0])
            I_vals = lut[I_idx]
            Q_vals = lut[Q_idx]
            norm = 3.0

        Ns_sym = bits_per_symbol * Ns
        I_full = np.repeat(I_vals, Ns_sym)
        Q_full = np.repeat(Q_vals, Ns_sym)
        
        t = make_time_axis(len(I_full), fs)
        arg = 2 * np.pi * fc * t + phi_ref
        c = np.cos(arg)
        sn = np.sin(arg)
        
        s = Ac * ((I_full / norm) * c + (Q_full / norm) * sn)
        
        sym_bits_interleaved = []
        for k in range(num_sym):
             chunk = []
             for j in range(bits_per_axis):
                 chunk.append(int(I_bits_mat[k, j]))
                 chunk.append(int(Q_bits_mat[k, j]))
             sym_bits_interleaved.append(chunk)

        meta.update({
            "axis_levels": axis_levels, "bits_per_axis": bits_per_axis,
            "bits_per_symbol": bits_per_symbol, "phi_ref": phi_ref,
            "pad_bits": pad, "symbols": num_sym,
            "sym_bits": sym_bits_interleaved,
            "I": I_vals.tolist(), "Q": Q_vals.tolist(),
            "norm": norm, "warnings": warnings,
        })
        return s, meta

    if scheme == "16QAM":
        bits_list, pad = _pad_bits(bits, 4)
        bits_arr = np.array(bits_list, dtype=int)
        
        num_sym = len(bits_arr) // 4
        sym_rows = bits_arr.reshape(num_sym, 4)
        
        powers = np.array([2, 1])
        I_idx = sym_rows[:, 0:2].dot(powers)
        Q_idx = sym_rows[:, 2:4].dot(powers)
        
        lut = np.array([-3.0, -1.0, 3.0, 1.0])
        I_vals = lut[I_idx]
        Q_vals = lut[Q_idx]
        norm = 3.0
        warnings += _warn_params(params, extra_freqs=[])

        Ns_sym = 4 * Ns
        I_full = np.repeat(I_vals, Ns_sym)
        Q_full = np.repeat(Q_vals, Ns_sym)
        
        t = make_time_axis(len(I_full), fs)
        c = np.cos(2 * np.pi * fc * t)
        sn = np.sin(2 * np.pi * fc * t)
        
        s = Ac * ((I_full / norm) * c + (Q_full / norm) * sn)
        
        sym_bits_tuples = [tuple(r) for r in sym_rows.tolist()]

        meta.update({
            "pad_bits": pad, "symbols": num_sym,
            "sym_bits": sym_bits_tuples,
            "I": I_vals.tolist(), "Q": Q_vals.tolist(),
            "norm": norm, "warnings": warnings,
        })
        return s, meta

    raise ValueError(f"Unknown modulation scheme: {scheme}")


# ----------------------------
# Demodulation (Vectorized)
# ----------------------------

def demodulate(s_t: np.ndarray, scheme: str, params: SimParams, **kwargs) -> Tuple[List[int], Dict[str, Any]]:
    scheme = scheme.upper()
    Ns = int(params.samples_per_bit)
    fs = float(params.fs)
    Ac = float(params.Ac)
    fc = float(params.fc)

    meta: Dict[str, Any] = {"scheme": scheme}
    warnings: List[str] = []
    
    # Global time axis
    N = len(s_t)
    t = make_time_axis(N, fs)

    if scheme == "ASK":
        A0 = float(kwargs.get("A0", 0.2))
        A1 = float(kwargs.get("A1", 1.0))
        thr = 0.5 * (A0 + A1)
        warnings += _warn_params(params, extra_freqs=[])
        
        I_corr, Q_corr = _vec_iq_correlator(s_t, t, fc, Ns)
        A_est = np.sqrt(I_corr**2 + Q_corr**2) / (Ac if Ac != 0 else 1.0)
        
        bits_out = np.where(A_est >= thr, 1, 0)
        
        meta.update({"A0": A0, "A1": A1, "thr": thr, 
                     "A_hat": A_est.tolist(), "warnings": warnings})
        return bits_out.tolist(), meta

    if scheme == "BFSK":
        if "f0" in kwargs and "f1" in kwargs:
            f0, f1 = float(kwargs["f0"]), float(kwargs["f1"])
            tone_sep = None
            # Ensure canonical order f0 < f1 to match Modulator
            if f1 < f0:
                f0, f1 = f1, f0
        else:
            Tb = float(params.Tb)
            tone_sep = float(kwargs.get("tone_sep", 2.0))
            f0 = fc - tone_sep / Tb
            f1 = fc + tone_sep / Tb
            if f1 < f0: f0, f1 = f1, f0
        
        warnings += _warn_params(params, extra_freqs=[f0, f1])

        # Correlate with f0 and f1 separately
        I0, Q0 = _vec_iq_correlator(s_t, t, f0, Ns)
        E0 = I0**2 + Q0**2
        
        I1, Q1 = _vec_iq_correlator(s_t, t, f1, Ns)
        E1 = I1**2 + Q1**2
        
        bits_out = np.where(E1 > E0, 1, 0)
        
        meta.update({
            "tone_sep": tone_sep, "f0": f0, "f1": f1,
            "E0": E0.tolist(), "E1": E1.tolist(), "warnings": warnings
        })
        return bits_out.tolist(), meta

    if scheme == "MFSK":
        L = int(kwargs.get("L", 2))
        fd = float(kwargs.get("fd", 1.0))
        M = 2 ** L
        Ns_sym = L * Ns
        
        freqs = [fc + (2 * (i + 1) - 1 - M) * fd for i in range(M)]
        warnings += _warn_params(params, extra_freqs=freqs)
        
        E_matrix = []
        for f in freqs:
            I, Q = _vec_iq_correlator(s_t, t, float(f), Ns_sym)
            E_matrix.append(I**2 + Q**2)
        
        E_matrix = np.column_stack(E_matrix)
        chosen_idx = np.argmax(E_matrix, axis=1)
        
        shifts = np.arange(L - 1, -1, -1)
        bits_mat = (chosen_idx[:, None] >> shifts) & 1
        bits_out = bits_mat.flatten()

        meta.update({
            "L": L, "M": M, "fd": fd, "freqs": freqs,
            "chosen_idx": chosen_idx.tolist(),
            "energies": E_matrix.tolist(), "warnings": warnings
        })
        return bits_out.tolist(), meta

    if scheme == "BPSK":
        warnings += _warn_params(params, extra_freqs=[])
        phase1 = float(kwargs.get("phase1", 0.0))
        phase0 = float(kwargs.get("phase0", np.pi))
        
        I_corr, Q_corr = _vec_iq_correlator(s_t, t, fc, Ns)
        
        I_hat = I_corr / (Ac if Ac != 0 else 1.0)
        
        phi = np.arctan2(-Q_corr, I_corr)
        
        def _vec_ang_dist(a, b):
            return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
            
        d1 = _vec_ang_dist(phi, phase1)
        d0 = _vec_ang_dist(phi, phase0)
        
        bits_out = np.where(d1 <= d0, 1, 0)
        
        meta.update({"phase1": phase1, "phase0": phase0, 
                     "I_hat": I_hat.tolist(), "warnings": warnings})
        return bits_out.tolist(), meta

    if scheme == "DPSK":
        warnings += _warn_params(params, extra_freqs=[])
        phase_init = float(kwargs.get("phase_init", 0.0))
        delta_phase = float(kwargs.get("delta_phase", np.pi))
        
        I_corr, Q_corr = _vec_iq_correlator(s_t, t, fc, Ns)
        
        phi_hat = np.arctan2(-Q_corr, I_corr)
        
        prev_phis = np.r_[phase_init, phi_hat[:-1]]
        dphi = (phi_hat - prev_phis + np.pi) % (2 * np.pi) - np.pi
        
        dist_change = np.abs(np.abs(dphi) - np.abs(delta_phase))
        dist_same = np.abs(dphi)
        
        bits_out = np.where(dist_change < dist_same, 1, 0)
        
        meta.update({
            "phase_init": phase_init, "delta_phase": delta_phase,
            "phi_hat": phi_hat.tolist(), "delta_hat": dphi.tolist(),
            "warnings": warnings
        })
        return bits_out.tolist(), meta

    if scheme == "QPSK":
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        Ns_sym = 2 * Ns
        warnings += _warn_params(params, extra_freqs=[])
        
        num_sym = len(s_t) // Ns_sym
        
        arg = 2 * np.pi * fc * t + phi_ref
        c = np.cos(arg)
        sn = np.sin(arg)
        
        prod_c = s_t * c
        prod_s = s_t * sn
        
        I_ints = prod_c.reshape(num_sym, Ns_sym).sum(axis=1) * (2.0 / Ns_sym)
        Q_ints = prod_s.reshape(num_sym, Ns_sym).sum(axis=1) * (2.0 / Ns_sym)
        
        den = (Ac if Ac != 0 else 1.0)
        I_sym = I_ints * np.sqrt(2.0) / den
        Q_sym = -Q_ints * np.sqrt(2.0) / den
        
        b0 = np.where(I_sym >= 0, 1, 0)
        b1 = np.where(Q_sym >= 0, 1, 0)
        
        bits_out = np.column_stack((b0, b1)).flatten()
        
        meta.update({
            "phi_ref": phi_ref, "symbols": num_sym,
            "I_hat": I_sym.tolist(), "Q_hat": Q_sym.tolist(),
            "warnings": warnings,
        })
        return bits_out.tolist(), meta

    if scheme == "QAM":
        axis_levels = int(kwargs.get("axis_levels", 2))
        
        # --- FIX: explicit validation to pass tests ---
        if axis_levels not in (2, 4):
            raise ValueError(f"QAM axis_levels must be 2 or 4, got {axis_levels}")
        # ---------------------------------------------
        
        phi_ref = float(kwargs.get("phi_ref", 0.0))
        norm = 1.0 if axis_levels == 2 else 3.0
        
        bits_per_symbol = 2 if axis_levels == 2 else 4
        Ns_sym = bits_per_symbol * Ns
        num_sym = len(s_t) // Ns_sym
        warnings += _warn_params(params, extra_freqs=[])
        
        arg = 2 * np.pi * fc * t + phi_ref
        c = np.cos(arg)
        sn = np.sin(arg)
        
        I_ints = (s_t * c).reshape(num_sym, Ns_sym).sum(axis=1) * (2.0 / Ns_sym)
        Q_ints = (s_t * sn).reshape(num_sym, Ns_sym).sum(axis=1) * (2.0 / Ns_sym)
        
        den = (Ac if Ac != 0 else 1.0)
        
        I_levels = I_ints * norm / den
        Q_levels = Q_ints * norm / den
        
        if axis_levels == 2:
            I_dec = np.where(I_levels >= 0, 1.0, -1.0)
            Q_dec = np.where(Q_levels >= 0, 1.0, -1.0)
            
            bI = np.where(I_dec > 0, 1, 0)
            bQ = np.where(Q_dec > 0, 1, 0)
            
            bits_out = np.column_stack((bI, bQ)).flatten()
            
        else:
            lut = np.array([-3.0, -1.0, 1.0, 3.0])
            I_dec = _nearest_level_vec(I_levels, lut)
            Q_dec = _nearest_level_vec(Q_levels, lut)
            
            I_idx = ((I_dec + 3) / 2).astype(int)
            Q_idx = ((Q_dec + 3) / 2).astype(int)
            
            bit_lut = np.array([[0,0], [0,1], [1,1], [1,0]])
            
            bI = bit_lut[I_idx]
            bQ = bit_lut[Q_idx]
            
            bits_out = np.column_stack((bI[:,0], bQ[:,0], bI[:,1], bQ[:,1])).flatten()

        meta.update({
            "axis_levels": axis_levels, "bits_per_axis": bits_per_symbol//2,
            "bits_per_symbol": bits_per_symbol, "phi_ref": phi_ref,
            "symbols": num_sym,
            "I_hat": I_levels.tolist(), "Q_hat": Q_levels.tolist(),
            "I_dec": I_dec.tolist(), "Q_dec": Q_dec.tolist(),
            "norm": norm, "warnings": warnings
        })
        return bits_out.tolist(), meta

    if scheme == "16QAM":
        Ns_sym = 4 * Ns
        num_sym = len(s_t) // Ns_sym
        norm = 3.0
        warnings += _warn_params(params, extra_freqs=[])
        
        I_ints, Q_ints = _vec_iq_correlator(s_t, t, fc, Ns_sym)
        den = (Ac if Ac != 0 else 1.0)
        I_levels = I_ints * norm / den
        Q_levels = Q_ints * norm / den
        
        lut = np.array([-3.0, -1.0, 1.0, 3.0])
        I_dec = _nearest_level_vec(I_levels, lut)
        Q_dec = _nearest_level_vec(Q_levels, lut)
        
        I_idx = ((I_dec + 3) / 2).astype(int)
        Q_idx = ((Q_dec + 3) / 2).astype(int)
        
        bit_lut = np.array([[0,0], [0,1], [1,1], [1,0]])
        bI = bit_lut[I_idx]
        bQ = bit_lut[Q_idx]
        
        bits_out = np.column_stack((bI[:,0], bI[:,1], bQ[:,0], bQ[:,1])).flatten()

        meta.update({
            "symbols": num_sym,
            "I_hat": I_levels.tolist(), "Q_hat": Q_levels.tolist(),
            "I_dec": I_dec.tolist(), "Q_dec": Q_dec.tolist(),
            "norm": norm, "warnings": warnings
        })
        return bits_out.tolist(), meta

    raise ValueError(f"Unknown demodulation scheme: {scheme}")


# ----------------------------
# End-to-end simulation wrapper
# ----------------------------

def simulate_d2a(bits: List[int], scheme: str, params: SimParams, **kwargs) -> SimResult:
    _validate_bits(bits)

    tx, meta_tx = modulate(bits, scheme, params, **kwargs)
    rx_bits, meta_rx = demodulate(tx, scheme, params, **kwargs)

    # Trim padding
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