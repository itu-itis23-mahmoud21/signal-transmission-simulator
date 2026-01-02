from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from scipy.signal import hilbert
from utils import SimParams, SimResult, make_time_axis
from a2d import gen_message

def am_modulate(m_t: np.ndarray, t: np.ndarray, Ac: float, fc: float, ka: float) -> np.ndarray:
    return (Ac + ka * m_t) * np.cos(2*np.pi*fc*t)


def fm_modulate(m_t: np.ndarray, t: np.ndarray, Ac: float, fc: float, kf: float, fs: float) -> np.ndarray:
    # phase = 2π fc t + 2π kf ∫ m(t) dt
    integral = np.cumsum(m_t) / fs
    phase = 2*np.pi*fc*t + 2*np.pi*kf*integral
    return Ac * np.cos(phase)


def pm_modulate(m_t: np.ndarray, t: np.ndarray, Ac: float, fc: float, kp: float) -> np.ndarray:
    phase = 2*np.pi*fc*t + kp * m_t
    return Ac * np.cos(phase)


def am_demodulate(s_t: np.ndarray) -> np.ndarray:
    analytic = hilbert(s_t)
    env = np.abs(analytic)
    # remove DC (carrier amplitude component)
    return env - np.mean(env)


def fm_demodulate(s_t: np.ndarray, fc: float, fs: float) -> np.ndarray:
    analytic = hilbert(s_t)
    phase = np.unwrap(np.angle(analytic))
    # instantaneous frequency: (1/2π) d/dt phase
    inst_freq = np.gradient(phase) * fs / (2*np.pi)
    # remove carrier and center
    return inst_freq - fc


def pm_demodulate(s_t: np.ndarray, fc: float, t: np.ndarray) -> np.ndarray:
    analytic = hilbert(s_t)
    phase = np.unwrap(np.angle(analytic))
    carrier_phase = 2*np.pi*fc*t
    return phase - carrier_phase


def simulate_a2a(
    kind: str,
    scheme: str,
    params: SimParams,
    *,
    Am: float,
    fm: float,
    duration: float,
    ka: float = 0.5,
    kf: float = 5.0,
    kp: float = 1.0,
) -> SimResult:
    fs = float(params.fs)
    fc = float(params.fc)
    Ac = float(params.Ac)

    N = int(duration * fs)
    t = make_time_axis(N, fs)
    m = gen_message(t, kind, Am, fm)

    meta: Dict[str, Any] = {"scheme": scheme, "kind": kind, "Am": Am, "fm": fm, "Ac": Ac, "fc": fc, "fs": fs}

    if scheme == "AM":
        s = am_modulate(m, t, Ac, fc, ka=ka)
        m_hat = am_demodulate(s)
        meta["ka"] = ka

    elif scheme == "FM":
        s = fm_modulate(m, t, Ac, fc, kf=kf, fs=fs)
        m_hat = fm_demodulate(s, fc=fc, fs=fs)
        meta["kf"] = kf

    elif scheme == "PM":
        s = pm_modulate(m, t, Ac, fc, kp=kp)
        m_hat = pm_demodulate(s, fc=fc, t=t)
        # remove DC offset
        m_hat = m_hat - np.mean(m_hat)
        meta["kp"] = kp

    else:
        raise ValueError("scheme must be AM, FM, or PM")

    return SimResult(
        t=t,
        signals={"m(t)": m, "tx": s, "recovered": m_hat},
        bits={},
        meta=meta
    )
