from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import hilbert

from utils import SimParams, SimResult, make_time_axis
from a2d import gen_message


# -----------------------------
# Helpers
# -----------------------------
def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _hilbert_reflect_center(x: np.ndarray, pad: int) -> np.ndarray:
    """
    Compute analytic signal using reflect-padding to reduce edge transients,
    then return only the original (center) segment.
    """
    n = int(x.size)
    if n == 0:
        return hilbert(x)

    pad = int(pad)
    pad = max(0, min(pad, n - 1))  # keep valid
    if pad == 0:
        return hilbert(x)

    xpad = np.pad(x, (pad, pad), mode="reflect")
    apad = hilbert(xpad)
    return apad[pad:-pad]


def _analytic_amp_phase(x: np.ndarray, *, pad: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return instantaneous amplitude and unwrapped phase via analytic signal.
    Uses optional reflect-padding to reduce boundary artifacts.
    """
    analytic = _hilbert_reflect_center(x, pad) if pad else hilbert(x)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    return amp, phase


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Moving average with reflect-padding to avoid edge artifacts.
    Ensures output length == input length and avoids "drop-to-zero" at boundaries.
    """
    if win <= 1:
        return x
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    win = int(win)

    # Prefer odd window for exact length match after reflect-padding + valid convolution
    if win % 2 == 0:
        win += 1

    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    k = np.ones(win, dtype=float) / float(win)
    y = np.convolve(xpad, k, mode="valid")  # length == len(x) for odd win
    return y


# -----------------------------
# Book-aligned modulators (Ch.16.1 conventions)
# -----------------------------
def am_modulate(x_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, na: float) -> np.ndarray:
    """
    Book AM (DSBTC / DSB-LC) form:
      s(t) = Ac * [1 + na * x(t)] * cos(2π f_c t)

    where x(t) is normalized to unit amplitude (max|x| = 1).
    """
    return float(Ac) * (1.0 + float(na) * x_t) * np.sin(2 * np.pi * float(fc) * t)


def pm_modulate(m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, kp: float) -> np.ndarray:
    """
    Phase modulation (PM):
      s(t) = Ac cos(2π f_c t + kp * m(t))
    """
    return float(Ac) * np.sin(2 * np.pi * float(fc) * t + float(kp) * m_t)


def fm_modulate(
    m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, kf: float, fs: float
) -> np.ndarray:
    """
    Frequency modulation (FM):
      s(t) = Ac cos( 2π f_c t + 2π kf ∫ m(τ) dτ )

    Here kf is a frequency sensitivity in (Hz per unit amplitude).
    Instantaneous frequency: f_i(t) = f_c + kf*m(t)
    """
    fs = float(fs)
    integral = np.cumsum(m_t) / fs  # ∫ m(t) dt (rectangular integration)
    phase = 2 * np.pi * float(fc) * t + 2 * np.pi * float(kf) * integral
    return float(Ac) * np.sin(phase)


# -----------------------------
# Simple "ideal" demodulators for visualization
# -----------------------------
def am_demodulate_envelope(
    s_t: np.ndarray, *, Ac: float, na: float, m_peak: float, fs: float, fc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Envelope detector (analytic-signal envelope).
    Returns: (m_hat, envelope_est)
    """
    pad = max(8, int(round(0.01 * fs)))  # ~10 ms padding
    env_est, _ = _analytic_amp_phase(s_t, pad=pad)

    win = max(1, int(round(fs / max(1.0, fc) * 1.0)))
    env_est = _moving_average(env_est, win)

    # stabilize edges (Hilbert envelope has boundary transients, worse for square)
    if env_est.size >= 2:
        env_est[0] = env_est[1]
        env_est[-1] = env_est[-2]

    # NEW: clamp a small guard region at both ends to kill residual spikes
    if env_est.size >= 10:
        guard = max(2, int(round(0.01 * fs)))      # ~10 ms
        guard = min(guard, env_est.size // 4)      # keep reasonable
        env_est[:guard] = env_est[guard]
        env_est[-guard:] = env_est[-guard - 1]

    na = float(na)
    Ac = float(Ac)

    if na == 0.0 or Ac == 0.0 or m_peak == 0.0:
        return np.zeros_like(s_t), env_est

    # Ideal envelope: Ac * (1 + na*x(t))
    x_hat = (env_est / Ac - 1.0) / na
    m_hat = x_hat * float(m_peak)   # de-normalize back to original message amplitude
    if m_hat.size >= 10:
        m_hat[:guard] = m_hat[guard]
        m_hat[-guard:] = m_hat[-guard - 1]
    return m_hat, env_est


def pm_demodulate_phase(
    s_t: np.ndarray, *, t: np.ndarray, fc: float, kp: float, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherent PM demod via analytic phase:
      φ_dev(t) ≈ unwrap(angle(hilbert(s))) - 2π f_c t
      m_hat(t) = φ_dev(t) / kp
    Returns: (m_hat, phase_dev)
    """
    pad = max(8, int(round(0.01 * fs)))
    _, phase = _analytic_amp_phase(s_t, pad=pad)
    phase_dev = phase - 2 * np.pi * float(fc) * t

    # remove residual offset/drift
    phase_dev = phase_dev - np.mean(phase_dev)

    kp = float(kp)
    if kp == 0.0:
        return np.zeros_like(s_t), phase_dev

    m_hat = phase_dev / kp
    return m_hat, phase_dev


def fm_demodulate_instfreq(
    s_t: np.ndarray, *, fc: float, kf: float, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FM demod via instantaneous frequency:
      f_i(t) = (1/2π) d/dt phase(t)
      m_hat(t) = (f_i(t) - f_c) / kf
    Returns: (m_hat, inst_freq)
    """
    pad = max(8, int(round(0.01 * fs)))
    _, phase = _analytic_amp_phase(s_t, pad=pad)
    inst_freq = np.gradient(phase) * float(fs) / (2 * np.pi)

    # stabilize derivative at the edges
    if inst_freq.size >= 2:
        inst_freq[0] = inst_freq[1]
        inst_freq[-1] = inst_freq[-2]

    # mild smoothing (derivative amplifies noise)
    win = max(1, int(round(fs / max(1.0, fc) * 0.10)))
    inst_freq = _moving_average(inst_freq, win)

    # stabilize edges (derivative + unwrap can still spike near boundaries)
    if inst_freq.size >= 2:
        inst_freq[0] = inst_freq[1]
        inst_freq[-1] = inst_freq[-2]

    # NEW: clamp a small guard region at both ends to kill residual boundary spikes
    if inst_freq.size >= 10:
        guard = max(2, int(round(0.005 * fs)))  # ~0.5% of a second worth of samples (e.g., 50 at fs=10k)
        guard = min(guard, inst_freq.size // 4)  # keep it reasonable

        inst_freq[:guard] = inst_freq[guard]
        inst_freq[-guard:] = inst_freq[-guard - 1]


    kf = float(kf)
    if kf == 0.0:
        return np.zeros_like(s_t), inst_freq

    f_dev = inst_freq - float(fc)
    f_dev = f_dev - np.mean(f_dev)
    m_hat = f_dev / kf
    return m_hat, inst_freq


# -----------------------------
# Simulation entry point
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
    kf: float = 5.0,
    kp: float = 1.0,
) -> SimResult:
    """
    Analog → Analog modulation simulation (Ch.16.1 conventions).

    Pipeline:
      1) Generate analog message m(t)
      2) Modulate (AM / FM / PM) to get s(t)
      3) "Ideal" demodulate for visualization (envelope / phase / inst. freq)
    """
    scheme = str(scheme).upper()
    if scheme not in ("AM", "FM", "PM"):
        raise ValueError("scheme must be AM, FM, or PM")

    fs = _require_positive("params.fs", float(params.fs))
    fc = _require_positive("params.fc", float(params.fc))
    Ac = _require_positive("params.Ac", float(params.Ac))

    Am = float(Am)
    fm = float(fm)
    duration = float(duration)

    if duration < 0:
        raise ValueError("duration must be >= 0")
    if fm <= 0:
        raise ValueError("fm must be > 0")

    # time axis
    N = int(round(duration * fs))
    t = make_time_axis(N, fs) if N > 0 else np.array([], dtype=float)

    # message
    m = gen_message(t, kind, Am, fm) if N > 0 else np.array([], dtype=float)
    carrier = np.sin(2 * np.pi * fc * t) if N > 0 else np.array([], dtype=float)

    meta: Dict[str, Any] = {
        "scheme": scheme,
        "kind": kind,
        "Am": float(Am),
        "fm": float(fm),
        "duration": float(duration),
        "Ac": float(Ac),
        "fc": float(fc),
        "fs": float(fs),
    }

    signals: Dict[str, np.ndarray] = {"m(t)": m, "carrier": carrier}

    m_peak = float(np.max(np.abs(m))) if m.size else 0.0

    if scheme == "AM":
        # Book convention: x(t) normalized to unit amplitude
        x = (m / m_peak) if (m_peak > 0 and N > 0) else np.zeros_like(m)

        s = am_modulate(x, t, Ac=Ac, fc=fc, na=na) if N > 0 else np.array([], dtype=float)
        m_hat, env_est = am_demodulate_envelope(s, Ac=Ac, na=na, m_peak=m_peak, fs=fs, fc=fc)

        # In book form, modulation index is exactly na (since max|x|=1)
        mu = abs(float(na))
        meta["am"] = {
            "na": float(na),
            "modulation_index_mu": float(mu),
            "overmodulated": bool(mu > 1.0),
            "envelope_min_theory": float(Ac * (1.0 - mu)),
            "envelope_max_theory": float(Ac * (1.0 + mu)),
            "bandwidth_hint_hz": float(2.0 * fm),  # DSBTC ≈ 2B, single-tone B≈fm
        }

        env_theory = (float(Ac) * (1.0 + float(na) * x)) if N > 0 else np.array([], dtype=float)

        signals.update(
            {
                "tx": s,
                "envelope_est": env_est,
                "envelope_theory": env_theory,
                "recovered": m_hat,
            }
        )

    elif scheme == "FM":
        s = fm_modulate(m, t, Ac=Ac, fc=fc, kf=kf, fs=fs) if N > 0 else np.array([], dtype=float)
        m_hat, inst_freq = fm_demodulate_instfreq(s, fc=fc, kf=kf, fs=fs)

        delta_f = abs(float(kf)) * m_peak
        beta = (delta_f / fm) if fm > 0 else float("inf")
        bw_carson = 2.0 * (delta_f + fm)

        meta["fm"] = {
            "kf_hz_per_unit": float(kf),
            "delta_f_max_hz": float(delta_f),
            "beta_index": float(beta),
            "bw_carson_hz": float(bw_carson),
        }

        signals.update({"tx": s, "inst_freq": inst_freq, "recovered": m_hat})

    else:  # PM
        s = pm_modulate(m, t, Ac=Ac, fc=fc, kp=kp) if N > 0 else np.array([], dtype=float)
        m_hat, phase_dev = pm_demodulate_phase(s, t=t, fc=fc, kp=kp, fs=fs)

        delta_phi = abs(float(kp)) * m_peak  # radians
        delta_f = abs(float(kp)) * m_peak * fm  # Hz (sine approx)
        bw_carson = 2.0 * (delta_f + fm)

        meta["pm"] = {
            "kp_rad_per_unit": float(kp),
            "delta_phi_max_rad": float(delta_phi),
            "delta_f_max_hz_sine_approx": float(delta_f),
            "bw_carson_hz_sine_approx": float(bw_carson),
        }

        signals.update({"tx": s, "phase_dev": phase_dev, "recovered": m_hat})

    # Small UI-friendly summary
    summary: Dict[str, Any] = {
        "scheme": scheme,
        "fs": float(fs),
        "fc": float(fc),
        "Ac": float(Ac),
        "fm": float(fm),
        "Am": float(Am),
        "duration": float(duration),
    }
    if scheme == "AM":
        summary.update(
            {
                "na": float(na),
                "mu": float(meta["am"]["modulation_index_mu"]),
                "BW_hint_Hz": float(meta["am"]["bandwidth_hint_hz"]),
            }
        )
    elif scheme == "FM":
        summary.update(
            {
                "kf": float(kf),
                "Δf_max_Hz": float(meta["fm"]["delta_f_max_hz"]),
                "β": float(meta["fm"]["beta_index"]),
                "BW_Carson_Hz": float(meta["fm"]["bw_carson_hz"]),
            }
        )
    else:
        summary.update(
            {
                "kp": float(kp),
                "Δφ_max_rad": float(meta["pm"]["delta_phi_max_rad"]),
                "Δf_max_Hz": float(meta["pm"]["delta_f_max_hz_sine_approx"]),
                "BW_Carson_Hz": float(meta["pm"]["bw_carson_hz_sine_approx"]),
            }
        )
    meta["summary"] = summary

    return SimResult(t=t, signals=signals, bits={}, meta=meta)
