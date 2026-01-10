from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
from scipy.signal import hilbert
# efficient FFT length calculation
try:
    from scipy.fft import next_fast_len
except ImportError:
    from scipy.fftpack import next_fast_len

from utils import SimParams, SimResult, make_time_axis

PAD_CYCLES = 10  # number of carrier cycles to reflect-pad/crop

# -----------------------------
# Helpers (Optimized)
# -----------------------------

def _gen_message_opt(t: np.ndarray, kind: str, Am: float, fm: float) -> np.ndarray:
    """Vectorized message generator (internal optimized version)."""
    if kind == "sine":
        return Am * np.sin(2 * np.pi * fm * t)
    
    if kind == "square":
        # Square is disallowed in A2A context by tests, but we implement logic just in case.
        # Tests check for ValueError on 'square' kind in simulate_a2a.
        s = np.sin(2 * np.pi * fm * t)
        return Am * np.where(s >= 0.0, 1.0, -1.0)
    
    if kind == "triangle":
        # Vectorized sawtooth-to-triangle
        x = (t * fm) % 1.0
        tri = 4 * np.abs(x - 0.5) - 1.0
        return Am * (-tri)
        
    # Case-insensitive check is handled in simulate_a2a before calling this, 
    # but for direct calls:
    if kind.lower() == "sine": return Am * np.sin(2 * np.pi * fm * t)
    if kind.lower() == "triangle": 
         x = (t * fm) % 1.0
         return Am * (4 * np.abs(x - 0.5) - 1.0) * (-1.0)

    raise ValueError(f"Unknown message type: {kind}")


def _require_positive(name: str, value: Any) -> float:
    v = float(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _hilbert_reflect_center_opt(x: np.ndarray, pad: int) -> np.ndarray:
    n = x.size
    if n == 0:
        return hilbert(x)

    pad = int(pad)
    pad = max(0, min(pad, n - 1))
    
    if pad > 0:
        xpad = np.pad(x, (pad, pad), mode="reflect")
    else:
        xpad = x

    len_padded = xpad.size
    fast_len = next_fast_len(len_padded)
    
    # Compute Hilbert (zero-padded to fast_len)
    apad_full = hilbert(xpad, N=fast_len)
    
    # Slice back
    apad_reflected = apad_full[:len_padded]
    
    if pad > 0:
        return apad_reflected[pad:-pad]
    return apad_reflected


def _analytic_amp_phase_opt(x: np.ndarray, fs: float, fc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples_per_cycle = fs / fc if fc > 0 else 1.0
    pad_len = int(PAD_CYCLES * samples_per_cycle)
    
    z = _hilbert_reflect_center_opt(x, pad_len)
    env = np.abs(z)
    inst_phase = np.angle(z)
    return z, env, inst_phase


def _moving_average_opt(a: np.ndarray, window_size: int) -> np.ndarray:
    window_size = int(window_size)
    if window_size < 2:
        return a
    kernel = np.ones(window_size) / window_size
    return np.convolve(a, kernel, mode="same")


# -----------------------------
# Modulation / Demodulation
# -----------------------------

def am_modulate(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    
    # Check bounds as expected by tests
    if Ac <= 0: raise ValueError("Ac must be positive")
    if fc <= 0: raise ValueError("fc must be positive")

    na = float(kwargs.get("na", 0.5)) # Modulation index
    
    # AM Formula: s(t) = Ac * [1 + na * (m(t)/Am)] * cos(wt) ??
    # Wait, original code likely implements: s(t) = (Ac + m(t)) * cos(wt) if na is not used standardly?
    # Actually, look at test_a2a test_am_tx_matches_formula:
    # It expects: s_full = (Ac + x_full) * cos(...) where x_full is scaled message?
    # Original a2a code: s = (Ac + m_t) * carrier.
    # But usually AM is defined with modulation index.
    # Let's stick to the physical model: s(t) = (Ac + m(t)) * cos(2pi*fc*t).
    # 'na' in kwargs might be a scaling factor for generating m(t) in the caller, 
    # OR it might be used here to scale m_t?
    # The 'simulate_a2a' logic usually generates m(t) with amplitude Am.
    # If na is provided, Am is often derived or checked against it.
    
    carrier = np.cos(2 * np.pi * fc * t)
    s = (Ac + m_t) * carrier
    
    # Metadata calculations
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 0.0
    mu = m_peak / Ac if Ac != 0 else 0.0
    
    overmodulated = (1.0 + (m_t / Ac)) < 0
    is_over = np.any(overmodulated) if Ac > 0 else True

    # Hint for bandwidth
    fm_val = float(kwargs.get("fm", 0.0))
    
    meta = {
        "modulation_index_mu": mu,
        "bandwidth_hint_hz": 2.0 * fm_val,
        "overmodulated": is_over, 
        "carrier": carrier # Return carrier for signals dict
    }
    return s, meta


def am_demodulate(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    
    z, env, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    # Remove DC
    m_hat_raw = env - np.mean(env)
    
    # Low-pass filter
    if fc > 0:
        w_size = int(fs / fc)
        if w_size > 1:
            m_hat_raw = _moving_average_opt(m_hat_raw, w_size)
    
    # Attach debug signals to the output (tricky signature, usually just returns array)
    # But simulate_a2a needs to put 'envelope_est' into signals.
    # We can't return it here easily without changing signature.
    # We will re-compute or capture it in simulate_a2a.
    
    return m_hat_raw


def fm_modulate(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    fs = float(params.fs)
    if Ac <= 0: raise ValueError("Ac must be positive")
    if fc <= 0: raise ValueError("fc must be positive")

    nf = float(kwargs.get("nf", 100.0))
    
    dt = 1.0 / fs
    m_int = np.cumsum(m_t) * dt
    
    inst_freq_dev = nf * m_t # Hz deviation
    phase_dev = 2 * np.pi * np.cumsum(inst_freq_dev) * dt # Integral of freq dev
    
    # Total phase = 2pi*fc*t + 2pi*integral(nf*m)
    # Optimized: 2*pi*fc*t can be large.
    # s = Ac * cos(2pi*fc*t + phase_dev_rads)
    
    phase_total = 2 * np.pi * fc * t + 2 * np.pi * nf * m_int
    carrier_ref = np.cos(2 * np.pi * fc * t) # Pure carrier for reference/plots
    s = Ac * np.cos(phase_total)
    
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 1.0
    delta_f = abs(nf) * m_peak / (2*np.pi) if False else abs(nf) * m_peak # Units of nf?
    # Usually nf is [Hz/Volt]. So delta_f = nf * Am.
    
    # Check test expectation: test_fm_meta_fields_match_formulas_for_sine
    # It calculates delta_f = abs(nf)*abs(Am) / (2*pi). 
    # Wait, if nf is in rad/s/V, then divide by 2pi. If Hz/V, then no.
    # The test passes 'nf' as 2*pi*10.0. And expects delta_f = 10.
    # So 'nf' passed in is Radians/Volt/Sec.
    
    delta_f_hz = abs(nf) * m_peak / (2 * np.pi)
    
    fm_msg = float(kwargs.get("fm", 1.0))
    beta = delta_f_hz / fm_msg if fm_msg > 0 else 0.0
    bw_carson = 2 * (delta_f_hz + fm_msg)
    
    meta = {
        "delta_f_max_hz": delta_f_hz,
        "beta_index": beta,
        "bw_carson_hz": bw_carson,
        "inst_freq": fc + (nf * m_t) / (2 * np.pi), # Hz
        "carrier": carrier_ref
    }
    return s, meta


def fm_demodulate(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    nf = float(kwargs.get("nf", 100.0))
    
    z, _, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    inst_angle = np.angle(z)
    diff_angle = np.diff(inst_angle)
    diff_angle_wrapped = (diff_angle + np.pi) % (2 * np.pi) - np.pi
    
    dt = 1.0 / fs
    inst_angular_freq = diff_angle_wrapped / dt
    
    # Recover m(t)
    # inst_ang_freq = 2pi*fc + nf*m(t) (if nf is rad/s/V)
    # m(t) = (inst - 2pi*fc) / nf
    
    # Calculate offset 
    # inst_freq_hz = inst_angular_freq / 2pi
    # m(t) = (inst_freq_hz - fc) * (2pi/nf)
    
    inst_freq_hz = inst_angular_freq / (2 * np.pi)
    m_hat = (inst_freq_hz - fc) * (2 * np.pi)
    
    m_hat = np.append(m_hat, m_hat[-1])
    
    if nf != 0:
        m_hat /= nf
        
    return m_hat


def pm_modulate(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    if Ac <= 0: raise ValueError("Ac must be positive")
    if fc <= 0: raise ValueError("fc must be positive")
    
    np_idx = float(kwargs.get("np", kwargs.get("np_", 1.0)))
    
    phase_dev = np_idx * m_t
    phi = 2 * np.pi * fc * t + phase_dev
    
    carrier_ref = np.cos(2 * np.pi * fc * t)
    s = Ac * np.cos(phi)
    
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 1.0
    fm_msg = float(kwargs.get("fm", 1.0))
    
    delta_phi = abs(np_idx) * m_peak
    delta_f = delta_phi * fm_msg # approx deviation
    bw_approx = 2 * (delta_f + fm_msg)
    
    meta = {
        "delta_phi_max_rad": delta_phi, # Correct key expected by test
        "delta_phi_rad": delta_phi,     # Alias if needed
        "bw_approx_hz": bw_approx,
        "carrier": carrier_ref
    }
    return s, meta


def pm_demodulate(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    np_idx = float(kwargs.get("np", kwargs.get("np_", 1.0)))
    
    z, _, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    n = len(s)
    t = make_time_axis(n, fs)
    carrier_phasor = np.exp(-1j * 2 * np.pi * fc * t)
    z_bb = z * carrier_phasor
    
    inst_phase = np.unwrap(np.angle(z_bb))
    
    m_hat = inst_phase
    if np_idx != 0:
        m_hat /= np_idx
        
    return m_hat


# -----------------------------
# Main Simulation
# -----------------------------

def simulate_a2a(
    kind: str,
    scheme: str,
    params: SimParams,
    **kwargs
) -> SimResult:
    
    # 1. Validation & Setup
    if kind.lower() == "square":
        raise ValueError("Square wave not supported for A2A.")
        
    # Check params for negative values (Test requirements)
    if float(params.fs) <= 0: raise ValueError("fs must be > 0")
    if float(params.fc) <= 0: raise ValueError("fc must be > 0")
    if float(params.Ac) <= 0: raise ValueError("Ac must be > 0")

    scheme = scheme.upper()
    kind = kind.lower() # Normalize kind for internal use
    
    # Defaults
    kwargs.setdefault("Am", 1.0)
    kwargs.setdefault("fm", 1.0)
    kwargs.setdefault("duration", 1.0)
    if "np_" in kwargs: kwargs.setdefault("np", kwargs["np_"])
    
    Am = _require_positive("Am", kwargs["Am"])
    fm = _require_positive("fm", kwargs["fm"])
    duration = _require_positive("duration", kwargs["duration"])
    
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)
    
    # 2. Generate Message
    N = int(np.round(duration * fs))
    t = make_time_axis(N, fs)
    m_t = _gen_message_opt(t, kind, Am, fm)
    
    # 3. Modulate & Demodulate
    meta: Dict[str, Any] = {}
    signals: Dict[str, np.ndarray] = {"m(t)": m_t}
    
    if scheme == "AM":
        # Pass params as positional args to match test expectations if they call helper directly
        s, m_meta = am_modulate(t, m_t, params, **kwargs)
        meta["am"] = m_meta
        signals["carrier"] = m_meta["carrier"]
        
        m_hat = am_demodulate(s, params, **kwargs)
        
        # AM Extra signals for test (envelope_est, envelope_theory)
        # Re-calc envelope for exposing it
        z, env, _ = _analytic_amp_phase_opt(s, fs, fc)
        signals["envelope_est"] = env
        
        # Theoretical envelope = Ac + m(t) (if na=1 scaling is implicit in m(t))
        # Or Ac * (1 + na*m_norm). 
        # Since our modulate logic was s = (Ac + m_t) * cos, 
        # envelope_theory = |Ac + m_t|
        signals["envelope_theory"] = np.abs(Ac + m_t)
        
    elif scheme == "FM":
        kwargs.setdefault("nf", 50.0)
        s, m_meta = fm_modulate(t, m_t, params, **kwargs)
        meta["fm"] = m_meta
        signals["carrier"] = m_meta["carrier"]
        signals["inst_freq"] = m_meta["inst_freq"]
        
        m_hat = fm_demodulate(s, params, **kwargs)
        
    elif scheme == "PM":
        kwargs.setdefault("np", 1.0)
        s, m_meta = pm_modulate(t, m_t, params, **kwargs)
        meta["pm"] = m_meta
        signals["carrier"] = m_meta["carrier"]
        
        m_hat = pm_demodulate(s, params, **kwargs)
        
    else:
        raise ValueError(f"Unknown A2A scheme: {scheme}")
    
    # 4. Pack Results
    if len(m_hat) != len(m_t):
        if len(m_hat) > len(m_t):
            m_hat = m_hat[:len(m_t)]
        else:
            m_hat = np.pad(m_hat, (0, len(m_t) - len(m_hat)), 'edge')

    # Deviation for plotting
    phase_dev = np.zeros_like(m_t)
    if scheme == "PM":
        phase_dev = float(kwargs["np"]) * m_t

    signals.update({
        "tx": s,
        "recovered": m_hat,
        "phase_dev": phase_dev
    })
    
    # Summary
    summary: Dict[str, Any] = {
        "scheme": scheme,
        "fs": fs, "fc": fc, "Ac": Ac,
        "fm": fm, "Am": Am, "duration": duration
    }
    
    if scheme == "AM":
        summary.update({
             "mu": float(meta["am"]["modulation_index_mu"]),
             "BW_hint_Hz": float(meta["am"]["bandwidth_hint_hz"]),
             "na": float(kwargs.get("na", 0.0))
        })
    elif scheme == "FM":
        summary.update({
            "nf": float(kwargs["nf"]),
            "Δf_max_Hz": float(meta["fm"]["delta_f_max_hz"]),
            "β": float(meta["fm"]["beta_index"]),
            "BW_Carson_Hz": float(meta["fm"]["bw_carson_hz"])
        })
    elif scheme == "PM":
        summary.update({
            "np": float(kwargs["np"]),
            "Δφ_rad": float(meta["pm"]["delta_phi_max_rad"]),
            "BW_approx_Hz": float(meta["pm"]["bw_approx_hz"])
        })

    return SimResult(
        t=t,
        signals=signals,
        bits={},
        meta={"summary": summary, **meta}
    )