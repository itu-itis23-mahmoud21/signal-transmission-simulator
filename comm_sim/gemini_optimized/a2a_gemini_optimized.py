from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import hilbert

# Optimization: Efficient FFT size calculation
try:
    from scipy.fft import next_fast_len
except ImportError:
    from scipy.fftpack import next_fast_len

from utils import SimParams, SimResult, make_time_axis
from a2d import gen_message

PAD_CYCLES = 10

# -----------------------------
# Optimized Helpers
# -----------------------------

def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v

def _hilbert_opt(x: np.ndarray) -> np.ndarray:
    """
    Optimized Hilbert transform using next_fast_len for FFT speedup.
    Standard scipy.signal.hilbert can be slow if len(x) is prime.
    """
    n = x.size
    if n == 0:
        return np.array([], dtype=x.dtype)
    
    # distinct optimization: find optimal FFT size (e.g. power of 2)
    fast_len = next_fast_len(n)
    
    # scipy.signal.hilbert supports N= argument to pad internally
    h = hilbert(x, N=fast_len)
    
    # crop back to original length
    return h[:n]

def _analytic_signal_padded(x: np.ndarray, pad: int) -> np.ndarray:
    """
    Computes analytic signal with reflection padding to minimize edge effects.
    Uses optimized Hilbert internally.
    """
    n = x.size
    if n == 0:
        return _hilbert_opt(x)

    pad = int(pad)
    pad = max(0, min(pad, n - 1))
    
    if pad > 0:
        x_pad = np.pad(x, (pad, pad), mode="reflect")
        z_pad = _hilbert_opt(x_pad)
        return z_pad[pad:-pad]
    else:
        return _hilbert_opt(x)

def _moving_average_opt(x: np.ndarray, win: int) -> np.ndarray:
    """
    Moving average with valid-mode convolution on reflected edges 
    to preserve length and avoid boundary zero-drop.
    """
    if win <= 1 or x.size == 0:
        return x
    
    win = int(win)
    if win % 2 == 0: win += 1 # prefer odd for symmetric centering
    
    pad = win // 2
    # Reflect pad
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    
    # Optimized: Pre-compute kernel sum for normalization
    # using 'valid' convolution keeps the size exact
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x_pad, kernel, mode="valid")


# -----------------------------
# Book-aligned Modulators (Stallings 16.1)
# -----------------------------
# Note: These must match a2a.py signatures exactly for tests to import and use them.

def am_modulate(x_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, na: float) -> np.ndarray:
    """s(t) = Ac * [1 + na * x(t)] * sin(2π f_c t)"""
    # Optimized: In-place multiplication where possible or efficient broadcasting
    carrier = np.sin(2 * np.pi * fc * t)
    return Ac * (1.0 + na * x_t) * carrier

def pm_modulate(m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, np_: float) -> np.ndarray:
    """s(t) = Ac * sin(2π f_c t + np * m(t))"""
    phase = 2 * np.pi * fc * t + np_ * m_t
    return Ac * np.sin(phase)

def fm_modulate(m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, nf: float, fs: float) -> np.ndarray:
    """
    s(t) = Ac * sin(2π f_c t + φ(t))
    φ(t) = ∫ n_f m(τ) dτ
    """
    # Optimized cumulative sum
    # Note: we use direct cumsum assuming constant dt=1/fs. 
    phi = np.cumsum(m_t) * (nf / fs) 
    phase = 2 * np.pi * fc * t + phi
    return Ac * np.sin(phase)

# -----------------------------
# Main Simulation
# -----------------------------

def simulate_a2a(
    kind: str,
    scheme: str,
    params: SimParams,
    *,
    Am: float,
    fm: float,
    duration: float,
    na: float = 0.5,
    nf: float = 2 * np.pi * 5.0,
    np_: float = 1.0,
) -> SimResult:

    scheme = scheme.upper()
    kind = kind.lower()
    
    # 1. Input Validation
    if kind == "square":
        raise ValueError("Square waveform is not supported in A2A.")
    if scheme not in ("AM", "FM", "PM"):
        raise ValueError(f"Unknown scheme: {scheme}")

    fs = _require_positive("fs", params.fs)
    fc = _require_positive("fc", params.fc)
    Ac = _require_positive("Ac", params.Ac)
    
    # Handle duration=0
    if duration == 0.0:
        return SimResult(
            t=np.array([]), signals={}, bits={}, 
            meta={"summary": {}, "scheme": scheme, "kind": kind}
        )
    _require_positive("duration", duration)
    if fm <= 0: raise ValueError("fm must be > 0")

    # 2. Time Axis Generation (Padded)
    # This logic matches a2a.py to ensure continuous signals at window boundaries
    N = int(round(duration * fs))
    padN = max(8, int(round(PAD_CYCLES * fs / max(1.0, fc)))) if N > 0 else 0
    N_full = N + 2 * padN
    
    if N_full > 0:
        # Vectorized time generation
        t_full = (np.arange(N_full, dtype=float) - padN) / fs
    else:
        t_full = np.array([], dtype=float)

    # 3. Message Generation
    m_full = gen_message(t_full, kind, Am, fm) if N_full > 0 else np.array([], dtype=float)
    
    # 4. Processing
    signals: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {
        "scheme": scheme, "kind": kind, "Am": Am, "fm": fm, 
        "duration": duration, "Ac": Ac, "fc": fc, "fs": fs
    }
    
    m_peak = np.max(np.abs(m_full)) if m_full.size > 0 else 0.0
    
    # Define slicing for result
    def crop(arr):
        return arr[padN : padN + N] if N > 0 else np.array([], dtype=float)

    if scheme == "AM":
        # Book: x(t) is normalized
        x_full = (m_full / m_peak) if m_peak > 0 else np.zeros_like(m_full)
        
        # Modulate
        s_full = am_modulate(x_full, t_full, Ac=Ac, fc=fc, na=na)
        
        # Demodulate (Envelope)
        # Using analytic signal magnitude
        z_full = _analytic_signal_padded(s_full, pad=padN)
        env_full = np.abs(z_full)
        
        # Recover
        # env ≈ Ac(1 + na*x) => x_hat = (env/Ac - 1)/na
        # m_hat = x_hat * m_peak
        if na != 0 and Ac != 0:
            x_hat_full = (env_full / Ac - 1.0) / na
            m_hat_full = x_hat_full * m_peak
        else:
            m_hat_full = np.zeros_like(env_full)

        # Crop and Store
        s = crop(s_full)
        m_hat = crop(m_hat_full)
        env_est = crop(env_full)
        
        # For theory consistency test
        x_crop = crop(x_full)
        env_theory = Ac * (1.0 + na * x_crop)
        
        signals.update({
            "tx": s,
            "envelope_est": env_est,
            "envelope_theory": env_theory,
            "recovered": m_hat
        })
        
        # AM Metadata
        mu = abs(float(na))
        meta["am"] = {
            "na": float(na),
            "modulation_index_mu": mu,
            "overmodulated": mu > 1.0,
            "bandwidth_hint_hz": 2.0 * fm
        }

    elif scheme == "FM":
        # Modulate
        s_full = fm_modulate(m_full, t_full, Ac=Ac, fc=fc, nf=nf, fs=fs)
        
        # Demodulate (Instantaneous Frequency)
        z_full = _analytic_signal_padded(s_full, pad=padN)
        inst_phase = np.unwrap(np.angle(z_full))
        
        # Gradient = d(Phase)/dn * fs
        inst_freq_full = np.gradient(inst_phase) * fs / (2 * np.pi)
        
        # Smooth
        win_len = max(1, int(round(fs / max(1.0, fc) * 0.10)))
        inst_freq_full = _moving_average_opt(inst_freq_full, win_len)
        
        # Recover: m_hat = (2π/nf) * (fi - fc)
        f_dev = inst_freq_full - fc
        f_dev -= np.mean(f_dev) # Remove DC offset
        
        if nf != 0:
            m_hat_full = (2 * np.pi * f_dev) / nf
        else:
            m_hat_full = np.zeros_like(f_dev)

        signals.update({
            "tx": crop(s_full),
            "inst_freq": crop(inst_freq_full),
            "recovered": crop(m_hat_full)
        })
        
        # FM Metadata
        delta_f = abs(nf) * m_peak / (2 * np.pi)
        meta["fm"] = {
            "delta_f_max_hz": delta_f,
            "beta_index": (delta_f / fm) if fm > 0 else float('inf'),
            "bw_carson_hz": 2.0 * (delta_f + fm)
        }

    elif scheme == "PM":
        # Modulate
        s_full = pm_modulate(m_full, t_full, Ac=Ac, fc=fc, np_=np_)
        
        # Demodulate (Phase)
        z_full = _analytic_signal_padded(s_full, pad=padN)
        phase_raw = np.unwrap(np.angle(z_full))
        
        # Remove carrier slope 2*pi*fc*t
        phase_dev_full = phase_raw - (2 * np.pi * fc * t_full)
        phase_dev_full -= np.mean(phase_dev_full) # center
        
        if np_ != 0:
            m_hat_full = phase_dev_full / np_
        else:
            m_hat_full = np.zeros_like(phase_dev_full)
            
        signals.update({
            "tx": crop(s_full),
            "phase_dev": crop(phase_dev_full),
            "recovered": crop(m_hat_full)
        })
        
        # PM Metadata
        delta_phi = abs(np_) * m_peak
        delta_f_equiv = delta_phi * fm
        meta["pm"] = {
            "delta_phi_max_rad": delta_phi,
            "delta_f_max_hz_sine_approx": delta_f_equiv,
            "bw_carson_hz_sine_approx": 2.0 * (delta_f_equiv + fm)
        }

    # Common Signals
    t = crop(t_full)
    m = crop(m_full)
    carrier = np.sin(2 * np.pi * fc * t) if t.size > 0 else np.array([])
    
    signals["m(t)"] = m
    signals["carrier"] = carrier
    
    # Summary for UI
    summary: Dict[str, Any] = {
        "scheme": scheme, "fs": fs, "fc": fc, "Ac": Ac,
        "fm": fm, "Am": Am, "duration": duration
    }
    
    if scheme == "AM":
        summary.update({
            "na": na, "mu": meta["am"]["modulation_index_mu"],
            "BW_hint_Hz": meta["am"]["bandwidth_hint_hz"]
        })
    elif scheme == "FM":
        summary.update({
            "nf": nf, "Δf_max_Hz": meta["fm"]["delta_f_max_hz"],
            "β": meta["fm"]["beta_index"],
            "BW_Carson_Hz": meta["fm"]["bw_carson_hz"]
        })
    elif scheme == "PM":
        summary.update({
            "np": np_, "Δφ_max_rad": meta["pm"]["delta_phi_max_rad"],
            "Δf_max_Hz": meta["pm"]["delta_f_max_hz_sine_approx"],
            "BW_Carson_Hz": meta["pm"]["bw_carson_hz_sine_approx"]
        })
        
    meta["summary"] = summary

    return SimResult(t=t, signals=signals, bits={}, meta=meta)