from __future__ import annotations

"""
A2A (Analog → Analog) modulation/demodulation — optimized.

This file is a *drop-in* implementation for the assignment's GPT-optimized variant.

Key optimization vs the original implementation:
  - The simulation already generates a padded record (extra carrier cycles) and
    then crops the requested window. We leverage that to avoid *additional*
    reflect-padding inside the demodulators.
  - Use FFT-friendly lengths for Hilbert transform (next_fast_len) to reduce FFT cost.
  - Use a fast moving-average smoother via cumulative sum (O(N)).

Public API is preserved:
  - PAD_CYCLES
  - am_modulate, fm_modulate, pm_modulate
  - simulate_a2a(...)
"""

from typing import Any, Dict

import numpy as np
from scipy.signal import hilbert

try:
    # SciPy >= 1.4
    from scipy.fft import next_fast_len as _next_fast_len  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for older SciPy
    from scipy.fftpack import next_fast_len as _next_fast_len  # type: ignore

from utils import SimParams, SimResult
from a2d import gen_message

PAD_CYCLES = 10  # number of carrier cycles used for padding/cropping (8–12 is a good range)


# -----------------------------
# Helpers
# -----------------------------
def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _hilbert_fast_center(x: np.ndarray) -> np.ndarray:
    """
    Fast analytic signal computation using an FFT-friendly length.

    We compute hilbert(x, N=next_fast_len(len(x))) and then take the center segment.
    Since we already run the simulation on a padded record and then crop a window
    away from the edges, the small difference vs hilbert(x) is acceptable and
    improves performance.
    """
    n = int(x.size)
    if n == 0:
        return hilbert(x)

    nfft = int(_next_fast_len(n))
    if nfft == n:
        return hilbert(x)

    # Zero-pad in frequency domain to a fast FFT size, then return the original segment.
    return hilbert(x, N=nfft)[:n]


def _moving_average_fast(x: np.ndarray, win: int) -> np.ndarray:
    """
    O(N) moving average with reflect padding (same-length output).
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        return x
    win = int(win)
    if win <= 1:
        return x

    # Prefer odd window for symmetric smoothing
    if win % 2 == 0:
        win += 1

    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    c = np.cumsum(xpad, dtype=float)
    c = np.concatenate(([0.0], c))
    y = (c[win:] - c[:-win]) / float(win)
    # y length == n
    return y


# -----------------------------
# Modulators (kept identical in signature)
# -----------------------------
def am_modulate(
    x_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, na: float
) -> np.ndarray:
    """
    AM (DSB-LC): s(t) = Ac * (1 + na*x(t)) * sin(2π f_c t)
    x(t) should be normalized to |x(t)|<=1 for typical use.
    """
    x_t = np.asarray(x_t, dtype=float)
    t = np.asarray(t, dtype=float)
    return float(Ac) * (1.0 + float(na) * x_t) * np.sin(2 * np.pi * float(fc) * t)


def fm_modulate(
    m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, nf: float, fs: float
) -> np.ndarray:
    """
    FM (phase accumulator):
      φ(t) = ∫ n_f m(τ) dτ
      s(t) = Ac * sin(2π f_c t + φ(t))
    Discrete-time: φ[n] ≈ (n_f/fs) * cumsum(m[n])
    """
    m_t = np.asarray(m_t, dtype=float)
    t = np.asarray(t, dtype=float)
    fs = _require_positive("fs", fs)

    if m_t.size == 0:
        return np.asarray(m_t, dtype=float)

    phi = (float(nf) / float(fs)) * np.cumsum(m_t, dtype=float)
    return float(Ac) * np.sin(2 * np.pi * float(fc) * t + phi)


def pm_modulate(
    m_t: np.ndarray, t: np.ndarray, *, Ac: float, fc: float, np_: float
) -> np.ndarray:
    """
    PM:
      s(t) = Ac * sin(2π f_c t + n_p m(t))
    """
    m_t = np.asarray(m_t, dtype=float)
    t = np.asarray(t, dtype=float)
    return float(Ac) * np.sin(2 * np.pi * float(fc) * t + float(np_) * m_t)


# -----------------------------
# Main simulation
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
    nf: float = 2 * np.pi * 5.0,   # rad/s per unit amplitude (book n_f)
    np_: float = 1.0,              # rad per unit amplitude (book n_p)
) -> SimResult:
    """
    Analog → Analog modulation simulation.

    Outputs:
      - t: time axis for the requested window [0, duration)
      - signals: includes m(t), carrier, tx, recovered, and scheme-specific signals
      - meta: includes summary + scheme-specific metadata

    Notes:
      - We generate a padded record (extra carrier cycles on both sides) and then crop.
        This reduces demod edge artifacts without extra reflect padding during demod.
    """
    scheme = str(scheme).upper()
    kind = str(kind).lower()

    if scheme not in {"AM", "FM", "PM"}:
        raise ValueError("scheme must be one of: AM, FM, PM")
    if kind not in {"sine", "triangle"}:
        raise ValueError("kind must be 'sine' or 'triangle'")

    fs = _require_positive("params.fs", params.fs)
    fc = _require_positive("params.fc", params.fc)
    Ac = float(params.Ac)

    Am = float(Am)
    fm = _require_positive("fm", fm)
    duration = float(duration)
    if duration < 0:
        raise ValueError("duration must be >= 0")

    # Window length (requested) and padded length
    N = int(round(duration * fs))
    padN = max(8, int(round(PAD_CYCLES * fs / max(1.0, fc)))) if N > 0 else 0
    N_full = N + 2 * padN

    if N_full > 0:
        t_full = (np.arange(N_full, dtype=float) - padN) / fs
        m_full = gen_message(t_full, kind, Am, fm)
        m_peak = float(np.max(np.abs(m_full))) if m_full.size else 0.0
        # Normalize message for AM (x(t) in [-1,1] under typical conditions)
        x_full = (m_full / m_peak) if m_peak > 0 else np.zeros_like(m_full)
    else:
        t_full = np.array([], dtype=float)
        m_full = np.array([], dtype=float)
        x_full = np.array([], dtype=float)
        m_peak = 0.0

    # Crop helper
    def _crop(arr: np.ndarray) -> np.ndarray:
        if N <= 0:
            return np.array([], dtype=float)
        return np.asarray(arr[padN:padN + N], dtype=float)

    # Requested-window signals
    t = _crop(t_full)
    m = _crop(m_full)
    carrier = np.sin(2 * np.pi * fc * t) if N > 0 else np.array([], dtype=float)

    signals: Dict[str, np.ndarray] = {"m(t)": m, "carrier": carrier}

    meta: Dict[str, Any] = {
        "scheme": scheme,
        "kind": kind,
        "Am": float(Am),
        "fm": float(fm),
        "duration": float(duration),
        "na": float(na),
        "nf": float(nf),
        "np_": float(np_),
        "fs": float(fs),
        "fc": float(fc),
        "Ac": float(Ac),
        "padN": int(padN),
        "PAD_CYCLES": int(PAD_CYCLES),
    }

    # Duration==0 short-circuit (keeps meta shape expectations)
    if N <= 0:
        meta["summary"] = {
            "scheme": scheme,
            "fs": float(fs),
            "fc": float(fc),
            "Ac": float(Ac),
            "fm": float(fm),
            "Am": float(Am),
            "duration": float(duration),
        }
        return SimResult(t=t, signals=signals, bits={}, meta=meta)

    # -----------------------------
    # Scheme-specific processing
    # -----------------------------
    if scheme == "AM":
        # Modulate on the padded record, then demod on padded record, then crop.
        s_full = am_modulate(x_full, t_full, Ac=Ac, fc=fc, na=na)
        z_full = _hilbert_fast_center(s_full)
        env_est_full = np.abs(z_full)

        # Avoid division by zero: na==0 means unmodulated carrier
        if abs(float(na)) < 1e-15:
            m_hat_full = np.zeros_like(m_full, dtype=float)
        else:
            m_hat_full = (env_est_full / (Ac if Ac != 0 else 1.0) - 1.0) / float(na)
            m_hat_full = m_hat_full * m_peak  # convert back to message units

        # Theory envelope in message units
        env_theory_full = float(Ac) * (1.0 + float(na) * x_full)

        s = _crop(s_full)
        env_est = _crop(env_est_full)
        env_theory = _crop(env_theory_full)
        m_hat = _crop(m_hat_full)

        # Metadata
        mu = abs(float(na)) * (m_peak if m_peak > 0 else 0.0)
        overmod = bool((mu > 1.0) if np.isfinite(mu) else False)
        bw_hint = 2.0 * float(fm)

        meta["am"] = {
            "na": float(na),
            "modulation_index_mu": float(mu),
            "overmodulated": overmod,
            "bandwidth_hint_hz": float(bw_hint),
        }

        signals.update(
            {
                "tx": s,
                "envelope_est": env_est,
                "envelope_theory": env_theory,
                "recovered": m_hat,
            }
        )

    elif scheme == "FM":
        s_full = fm_modulate(m_full, t_full, Ac=Ac, fc=fc, nf=nf, fs=fs)

        # Inst. freq via analytic phase derivative (no extra reflect pad; rely on padN crop)
        z_full = _hilbert_fast_center(s_full)
        phase = np.unwrap(np.angle(z_full))

        # Forward difference is fast; we later smooth and crop away edges.
        inst_freq_full = np.diff(phase, prepend=phase[0]) * (fs / (2.0 * np.pi))

        # Stabilize endpoints
        if inst_freq_full.size >= 2:
            inst_freq_full[0] = inst_freq_full[1]
            inst_freq_full[-1] = inst_freq_full[-2]

        # Smooth over ~10% of a carrier period (same heuristic as original)
        win = max(1, int(round(fs / max(1.0, fc) * 0.10)))
        inst_freq_full = _moving_average_fast(inst_freq_full, win)

        if inst_freq_full.size >= 2:
            inst_freq_full[0] = inst_freq_full[1]
            inst_freq_full[-1] = inst_freq_full[-2]

        # Center frequency deviation so mean(inst_freq) ≈ fc
        f_dev = inst_freq_full - fc
        if f_dev.size:
            f_dev = f_dev - float(np.mean(f_dev))
        inst_freq_full = f_dev + fc

        if abs(float(nf)) < 1e-15:
            m_hat_full = np.zeros_like(m_full, dtype=float)
        else:
            m_hat_full = (2.0 * np.pi / float(nf)) * f_dev

        s = _crop(s_full)
        inst_freq = _crop(inst_freq_full)
        m_hat = _crop(m_hat_full)

        # Metadata (Carson's rule)
        delta_f = abs(float(nf)) * m_peak / (2.0 * np.pi) if m_peak > 0 else 0.0
        beta = (delta_f / float(fm)) if float(fm) != 0 else float("inf")
        bw_carson = 2.0 * (delta_f + float(fm))

        meta["fm"] = {
            "nf_rad_per_s_per_unit": float(nf),
            "delta_f_max_hz": float(delta_f),
            "beta_index": float(beta),
            "bw_carson_hz": float(bw_carson),
        }

        signals.update(
            {
                "tx": s,
                "inst_freq": inst_freq,
                "recovered": m_hat,
            }
        )

    else:  # PM
        s_full = pm_modulate(m_full, t_full, Ac=Ac, fc=fc, np_=np_)
        z_full = _hilbert_fast_center(s_full)
        phase = np.unwrap(np.angle(z_full))

        # Remove carrier phase; mean-center to eliminate constant offset
        phase_dev_full = phase - (2.0 * np.pi * fc * t_full)
        if phase_dev_full.size:
            phase_dev_full = phase_dev_full - float(np.mean(phase_dev_full))

        if abs(float(np_)) < 1e-15:
            m_hat_full = np.zeros_like(m_full, dtype=float)
        else:
            m_hat_full = phase_dev_full / float(np_)

        s = _crop(s_full)
        phase_dev = _crop(phase_dev_full)
        m_hat = _crop(m_hat_full)

        # Metadata (sine approximation / Carson-style)
        delta_phi = abs(float(np_)) * m_peak if m_peak > 0 else 0.0
        delta_f = delta_phi * float(fm)
        bw_carson = 2.0 * (delta_f + float(fm))

        meta["pm"] = {
            "np_rad_per_unit": float(np_),
            "delta_phi_max_rad": float(delta_phi),
            "delta_f_max_hz_sine_approx": float(delta_f),
            "bw_carson_hz_sine_approx": float(bw_carson),
        }

        signals.update(
            {
                "tx": s,
                "phase_dev": phase_dev,
                "recovered": m_hat,
            }
        )

    # Summary (common)
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
                "overmodulated": bool(meta["am"]["overmodulated"]),
                "bandwidth_hint_hz": float(meta["am"]["bandwidth_hint_hz"]),
            }
        )
    elif scheme == "FM":
        summary.update(
            {
                "nf": float(nf),
                "delta_f_max_hz": float(meta["fm"]["delta_f_max_hz"]),
                "beta_index": float(meta["fm"]["beta_index"]),
                "bw_carson_hz": float(meta["fm"]["bw_carson_hz"]),
            }
        )
    else:
        summary.update(
            {
                "np_": float(np_),
                "delta_phi_max_rad": float(meta["pm"]["delta_phi_max_rad"]),
                "delta_f_max_hz_sine_approx": float(meta["pm"]["delta_f_max_hz_sine_approx"]),
                "bw_carson_hz_sine_approx": float(meta["pm"]["bw_carson_hz_sine_approx"]),
            }
        )

    meta["summary"] = summary
    return SimResult(t=t, signals=signals, bits={}, meta=meta)
