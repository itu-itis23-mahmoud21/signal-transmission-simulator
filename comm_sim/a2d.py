from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from utils import SimParams, SimResult, make_time_axis
from d2d import line_encode


def gen_message(t: np.ndarray, kind: str, Am: float, fm: float) -> np.ndarray:
    if kind == "sine":
        return Am * np.sin(2*np.pi*fm*t)
    if kind == "square":
        return Am * np.sign(np.sin(2*np.pi*fm*t))
    if kind == "triangle":
        # triangle via sawtooth-like formula
        x = (t * fm) % 1.0
        tri = 4*np.abs(x - 0.5) - 1.0
        return Am * (-tri)
    raise ValueError(f"Unknown message type: {kind}")


def pcm_encode(m_t: np.ndarray, fs: float, n_bits: int, vmin: float, vmax: float) -> Tuple[List[int], Dict[str, Any]]:
    L = 2**n_bits
    # Uniform quantization in [vmin, vmax]
    x = np.clip(m_t, vmin, vmax)
    delta = (vmax - vmin) / L
    # indices 0..L-1
    idx = np.floor((x - vmin) / delta).astype(int)
    idx = np.clip(idx, 0, L-1)
    # quantized reconstruction value (mid-rise)
    q = vmin + (idx + 0.5) * delta

    # encode indices to bits
    bitstream: List[int] = []
    codewords: List[str] = []
    for k in idx.tolist():
        cw = format(k, f"0{n_bits}b")
        codewords.append(cw)
        bitstream.extend([1 if c == "1" else 0 for c in cw])

    meta = {
        "n_bits": n_bits,
        "L": L,
        "delta": delta,
        "idx": idx,
        "q": q,
        "codewords": codewords,
        "vmin": vmin,
        "vmax": vmax,
    }
    return bitstream, meta


def dm_encode(m_t: np.ndarray, delta: float) -> Tuple[List[int], Dict[str, Any]]:
    # Simple delta modulation: compare to staircase estimate
    est = 0.0
    bits: List[int] = []
    stairs = []
    for x in m_t:
        b = 1 if x >= est else 0
        bits.append(b)
        est += delta if b == 1 else -delta
        stairs.append(est)
    meta = {"delta": delta, "stair": np.array(stairs, dtype=float)}
    return bits, meta


def simulate_a2d(
    kind: str,
    technique: str,
    params: SimParams,
    *,
    Am: float,
    fm: float,
    duration: float,
    fs_mult: int = 8,
    pcm_nbits: int = 4,
    dm_delta: float = 0.1,
    linecode_scheme: str = "NRZ-L",
) -> SimResult:
    # Display sample rate (high enough for smooth waveform)
    fs_display = max(2000.0, 200.0 * fm)
    N = int(duration * fs_display)
    t = make_time_axis(N, fs_display)
    m = gen_message(t, kind, Am, fm)

    # Sampling for codec
    fs_samp = fs_mult * fm
    if fs_samp <= 0:
        fs_samp = 8 * fm
    step = max(1, int(round(fs_display / fs_samp)))
    samp_idx = np.arange(0, N, step, dtype=int)
    m_s = m[samp_idx]
    t_s = t[samp_idx]

    meta: Dict[str, Any] = {
        "kind": kind,
        "technique": technique,
        "fs_display": fs_display,
        "fs_samp": fs_display / step,
        "fs_mult": fs_mult,
        "samples": len(m_s),
    }

    if technique == "PCM":
        bits, m_pcm = pcm_encode(m_s, fs=meta["fs_samp"], n_bits=pcm_nbits, vmin=-Am, vmax=Am)
        meta["pcm"] = m_pcm
        q = m_pcm["q"]
        stair = None

    elif technique == "DM":
        bits, m_dm = dm_encode(m_s, delta=dm_delta)
        meta["dm"] = m_dm
        q = None
        stair = m_dm["stair"]

    else:
        raise ValueError("technique must be PCM or DM")

    # Line-code the produced bitstream for display
    # We reuse params.samples_per_bit for line-code sampling density; we use a separate fs for this segment
    # Create line-code params for bit display
    Ns = int(params.samples_per_bit)
    Tb = float(params.Tb)
    fs_bits = Ns / Tb
    lc_params = SimParams(fs=fs_bits, Tb=Tb, samples_per_bit=Ns, Ac=params.Ac, fc=params.fc)

    line_wave, lc_meta = line_encode(bits, linecode_scheme, lc_params)
    t_bits = make_time_axis(len(line_wave), fs_bits)

    meta["linecode"] = {"scheme": linecode_scheme, **lc_meta, "bit_len": len(bits)}

    signals = {
        "m(t)": m,
        "linecode": line_wave,
    }
    # For steps visualization
    meta["sampled"] = {"t_s": t_s, "m_s": m_s}
    if q is not None:
        meta["quantized"] = {"q": q}
    if stair is not None:
        meta["stair"] = {"stair": stair}

    return SimResult(
        t=t,  # main time axis for analog display
        signals=signals,
        bits={"bitstream": bits},
        meta={**meta, "t_bits": t_bits}
    )
