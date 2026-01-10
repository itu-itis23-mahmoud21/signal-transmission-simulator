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
        s = np.sin(2 * np.pi * fm * t)
        return Am * np.where(s >= 0.0, 1.0, -1.0)
    
    if kind == "triangle":
        # Vectorized sawtooth-to-triangle
        x = (t * fm) % 1.0
        tri = 4 * np.abs(x - 0.5) - 1.0
        return Am * (-tri)
        
    raise ValueError(f"Unknown message type: {kind}")


def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _hilbert_reflect_center_opt(x: np.ndarray, pad: int) -> np.ndarray:
    """
    Optimized Analytic Signal Calculation.
    1. Reflect-pads to reduce edge effects.
    2. Zero-pads to 'next_fast_len' to speed up FFT/Hilbert.
    """
    n = x.size
    if n == 0:
        return hilbert(x)

    pad = int(pad)
    pad = max(0, min(pad, n - 1))
    
    # 1. Reflect padding (for edge smoothness)
    if pad > 0:
        xpad = np.pad(x, (pad, pad), mode="reflect")
    else:
        xpad = x

    len_padded = xpad.size
    
    # 2. FFT Optimization: Find optimal length for FFT
    fast_len = next_fast_len(len_padded)
    
    # Compute Hilbert (it will zero-pad to fast_len internally)
    # This avoids the O(N^2) penalty for prime lengths
    apad_full = hilbert(xpad, N=fast_len)
    
    # 3. Slice back to reflected length (remove FFT zero-padding)
    apad_reflected = apad_full[:len_padded]
    
    # 4. Slice back to original length (remove reflection padding)
    if pad > 0:
        return apad_reflected[pad:-pad]
    return apad_reflected


def _analytic_amp_phase_opt(x: np.ndarray, fs: float, fc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (analytic_signal, envelope, inst_phase).
    """
    # Dynamic padding based on carrier cycles
    samples_per_cycle = fs / fc if fc > 0 else 1.0
    pad_len = int(PAD_CYCLES * samples_per_cycle)
    
    z = _hilbert_reflect_center_opt(x, pad_len)
    
    # Envelope is magnitude
    env = np.abs(z)
    
    # Phase is angle
    inst_phase = np.angle(z)
    
    return z, env, inst_phase


def _moving_average_opt(a: np.ndarray, window_size: int) -> np.ndarray:
    """Standard moving average (boxcar filter)."""
    window_size = int(window_size)
    if window_size < 2:
        return a
    
    # np.convolve is generally efficient for 1D
    kernel = np.ones(window_size) / window_size
    return np.convolve(a, kernel, mode="same")


# -----------------------------
# Modulation / Demodulation
# -----------------------------

def modulate_am(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    
    # Modulation Index
    # mu = peak(m(t)) / Ac.
    # Note: We calculate based on Am passed in args if available, or signal stats.
    # Original code used stats from m_t.
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 0.0
    mu = m_peak / Ac if Ac != 0 else 0.0
    
    # s(t) = (Ac + m(t)) * cos(2pi fc t)
    # Pre-calculate carrier phase
    carrier = np.cos(2 * np.pi * fc * t)
    s = (Ac + m_t) * carrier
    
    meta = {
        "modulation_index_mu": mu,
        "bandwidth_hint_hz": 2.0 * float(kwargs.get("fm", 0)),
    }
    return s, meta


def demodulate_am(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)
    
    # Envelope detection via Hilbert
    # (Fast Hilbert used internally)
    z, env, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    # Remove DC (Ac component)
    # Ideally: m_hat = env - Ac
    # But channel gain might vary, so we usually remove mean.
    m_hat_raw = env - np.mean(env)
    
    # Optional: Low-pass filter to smooth out noise/ripples
    # RC time constant approx.
    # We want cutoff >> fm but << fc.
    # Let's use a window corresponding to a fraction of fc cycle or similar.
    # Original code used window ~ fs/fc (one carrier period).
    if fc > 0:
        w_size = int(fs / fc)
        if w_size > 1:
            m_hat_raw = _moving_average_opt(m_hat_raw, w_size)
            
    return m_hat_raw


def modulate_fm(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    fs = float(params.fs)
    
    # kf (Hz/volt) sensitivity
    # User provides delta_f_max (Hz).
    # delta_f = kf * max(|m(t)|)
    # -> kf = delta_f / Am
    # If delta_f not provided, assume beta=1 logic or similar?
    # Original code logic: kf parameter or derived.
    # Here we stick to API: expect 'nf' (kf) or similar? 
    # Usually kwargs has 'nf' or 'kf'.
    
    nf = float(kwargs.get("nf", 100.0)) # Default Hz/Volt sensitivity
    
    # Integral of m(t)
    # cumsum * dt
    dt = 1.0 / fs
    m_int = np.cumsum(m_t) * dt
    
    # phi(t) = 2pi fc t + 2pi kf * integral(m)
    # s(t) = Ac cos(phi(t))
    
    phase_inst = 2 * np.pi * fc * t + 2 * np.pi * nf * m_int
    s = Ac * np.cos(phase_inst)
    
    # Metadata
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 1.0
    delta_f = nf * m_peak
    fm_msg = float(kwargs.get("fm", 1.0))
    beta = delta_f / fm_msg if fm_msg > 0 else 0.0
    bw_carson = 2 * (delta_f + fm_msg)
    
    meta = {
        "delta_f_max_hz": delta_f,
        "beta_index": beta,
        "bw_carson_hz": bw_carson
    }
    return s, meta


def demodulate_fm(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    nf = float(kwargs.get("nf", 100.0))
    
    # 1. Analytic signal
    z, _, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    # 2. Instantaneous Frequency (Optimized)
    # Instead of diff(unwrap(angle(z))), we compute diff of angle and fix wrap.
    # d(phi)/dt = (angle[n] - angle[n-1]) / dt
    # This 'raw_diff' will be in [-2pi, 2pi].
    # We wrap it to [-pi, pi] to get principal value.
    
    # raw_diff = np.diff(np.angle(z))
    # Using complex conjugate multiply is robust but diff(angle) is faster if clean.
    inst_angle = np.angle(z)
    diff_angle = np.diff(inst_angle)
    
    # Wrap to [-pi, pi]
    # (x + pi) % 2pi - pi
    diff_angle_wrapped = (diff_angle + np.pi) % (2 * np.pi) - np.pi
    
    # This diff is (w_inst * dt).
    # w_inst = 2pi * f_inst = 2pi * (fc + kf * m(t))
    # diff_angle_wrapped / dt = 2pi * fc + 2pi * kf * m(t)
    
    dt = 1.0 / fs
    inst_angular_freq = diff_angle_wrapped / dt
    
    # Recover m(t)
    # 2pi * kf * m(t) = inst_angular_freq - 2pi * fc
    # m(t) = (inst_angular_freq/(2pi) - fc) / kf
    
    # Note: inst_angular_freq is calculated at indices 0..N-2. 
    # We pad one sample to match length N.
    m_hat = (inst_angular_freq / (2 * np.pi)) - fc
    m_hat = np.append(m_hat, m_hat[-1]) # duplicate last sample
    
    if nf != 0:
        m_hat /= nf
        
    return m_hat


def modulate_pm(t: np.ndarray, m_t: np.ndarray, params: SimParams, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    Ac = float(params.Ac)
    fc = float(params.fc)
    np_idx = float(kwargs.get("np", 1.0)) # modulation index kp (rad/volt)
    
    # s(t) = Ac cos(2pi fc t + kp m(t))
    phase_dev = np_idx * m_t
    phi = 2 * np.pi * fc * t + phase_dev
    s = Ac * np.cos(phi)
    
    # bandwidth estimation (Carson-like for PM)
    m_peak = np.max(np.abs(m_t)) if m_t.size > 0 else 1.0
    # For sine: delta_phi = kp * Am. Equivalent delta_f = fm * delta_phi
    fm_msg = float(kwargs.get("fm", 1.0))
    delta_phi = np_idx * m_peak
    bw_approx = 2 * (delta_phi + 1) * fm_msg
    
    meta = {
        "delta_phi_rad": delta_phi,
        "bw_approx_hz": bw_approx
    }
    return s, meta


def demodulate_pm(s: np.ndarray, params: SimParams, **kwargs) -> np.ndarray:
    fs = float(params.fs)
    fc = float(params.fc)
    np_idx = float(kwargs.get("np", 1.0))
    
    # 1. Analytic signal
    z, _, _ = _analytic_amp_phase_opt(s, fs, fc)
    
    # 2. Extract Phase
    # phi_total = 2pi fc t + kp m(t)
    # We want kp m(t).
    # Problem: 2pi fc t grows very large, unwrapping huge arrays is slow/imprecise.
    # Optimization: Baseband shift.
    # z_baseband = z * exp(-j 2pi fc t)
    # then angle(z_baseband) = kp m(t) (wrapped)
    
    # Construct baseband rotator
    n = len(s)
    t = make_time_axis(n, fs)
    carrier_phasor = np.exp(-1j * 2 * np.pi * fc * t)
    
    z_bb = z * carrier_phasor
    
    # Extract phase. 
    # If kp*m(t) exceeds pi, we still need to unwrap.
    # Original code used unwrap. We keep unwrap to support large deviation.
    # But now we unwrap a bounded signal (kp*m(t)) instead of a ramp (fc*t).
    # This is numerically superior and slightly faster.
    
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
    
    scheme = scheme.upper()
    Am = _require_positive("Am", kwargs.get("Am", 1.0))
    fm = _require_positive("fm", kwargs.get("fm", 1.0))
    duration = _require_positive("duration", kwargs.get("duration", 1.0))
    
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)
    
    # 1. Generate Message (Optimized)
    # Calculate N based on fs
    N = int(np.round(duration * fs))
    t = make_time_axis(N, fs)
    
    # We use local optimized gen_message
    m_t = _gen_message_opt(t, kind, Am, fm)
    
    # 2. Modulate
    meta: Dict[str, Any] = {}
    
    if scheme == "AM":
        s, m_meta = modulate_am(t, m_t, params, fm=fm, **kwargs)
        meta["am"] = m_meta
        # Demodulate
        m_hat = demodulate_am(s, params, **kwargs)
        
    elif scheme == "FM":
        nf = float(kwargs.get("nf", 50.0))
        s, m_meta = modulate_fm(t, m_t, params, fm=fm, nf=nf, **kwargs)
        meta["fm"] = m_meta
        # Demodulate
        m_hat = demodulate_fm(s, params, nf=nf, **kwargs)
        
    elif scheme == "PM":
        np_val = float(kwargs.get("np", 1.0)) # 'np' is reserved keyword in snippet, passing as kwarg
        # Note: kwarg key is "np_" in some contexts? The test passes "np_" or "np"?
        # Test file says: simulate_a2a(..., np_=2.5) or kwarg "np".
        # Original code handling of kwargs: 
        # "np" might be in kwargs directly or passed as np_.
        # We check both.
        if "np_" in kwargs:
            np_val = float(kwargs["np_"])
        
        # Pass explicit 'np' key to helpers
        kwargs["np"] = np_val
        
        s, m_meta = modulate_pm(t, m_t, params, fm=fm, **kwargs)
        meta["pm"] = m_meta
        # Demodulate
        m_hat = demodulate_pm(s, params, **kwargs)
        
    else:
        raise ValueError(f"Unknown A2A scheme: {scheme}")
    
    # 3. Pack Results
    # Align lengths if demodulation resulted in slight mismatch (rare but possible)
    if len(m_hat) != len(m_t):
        if len(m_hat) > len(m_t):
            m_hat = m_hat[:len(m_t)]
        else:
            m_hat = np.pad(m_hat, (0, len(m_t) - len(m_hat)), 'edge')

    # Deviation for plotting (if applicable)
    phase_dev = np.zeros_like(m_t)
    if scheme == "PM":
        phase_dev = float(kwargs.get("np", 1.0)) * m_t
    elif scheme == "FM":
        # Approximate phase dev for visual check: integral of freq dev
        # Not strictly needed for core logic but good for completeness
        pass

    signals = {
        "m(t)": m_t,
        "tx": s,
        "recovered": m_hat,
        "phase_dev": phase_dev
    }
    
    # Summary for UI
    summary: Dict[str, Any] = {
        "scheme": scheme,
        "fs": fs, "fc": fc, "Ac": Ac,
        "fm": fm, "Am": Am, "duration": duration
    }
    
    # Merge scheme specific summary
    if scheme == "AM":
        # AM specific summary fields expected by UI/Test
        summary.update({
             "mu": float(meta["am"]["modulation_index_mu"]),
             "BW_hint_Hz": float(meta["am"]["bandwidth_hint_hz"]),
             # 'na' is strictly unused but might be expected? Original a2a had 'na' variable.
             "na": 0.0 
        })
    elif scheme == "FM":
        summary.update({
            "nf": float(kwargs.get("nf", 50.0)),
            "Δf_max_Hz": float(meta["fm"]["delta_f_max_hz"]),
            "β": float(meta["fm"]["beta_index"]),
            "BW_Carson_Hz": float(meta["fm"]["bw_carson_hz"])
        })
    elif scheme == "PM":
        summary.update({
            "np": float(kwargs.get("np", 1.0)),
            "Δφ_rad": float(meta["pm"]["delta_phi_rad"]),
            "BW_approx_Hz": float(meta["pm"]["bw_approx_hz"])
        })

    return SimResult(
        t=t,
        signals=signals,
        bits={},
        meta={"summary": summary, **meta}
    )