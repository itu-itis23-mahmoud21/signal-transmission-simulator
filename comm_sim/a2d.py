from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from utils import SimParams, SimResult, make_time_axis
from d2d import line_encode, line_decode


# -------------------------
# Message generation (analog)
# -------------------------

def gen_message(t: np.ndarray, kind: str, Am: float, fm: float) -> np.ndarray:
    if kind == "sine":
        return Am * np.sin(2 * np.pi * fm * t)

    if kind == "square":
        s = np.sin(2 * np.pi * fm * t)
        return Am * np.where(s >= 0.0, 1.0, -1.0)

    if kind == "triangle":
        # triangle via sawtooth-like formula
        x = (t * fm) % 1.0
        tri = 4 * np.abs(x - 0.5) - 1.0
        return Am * (-tri)

    raise ValueError(f"Unknown message type: {kind}")


def _choose_display_fs(fm: float, fs_samp: float) -> float:
    # Make the analog plot smooth AND also show the sample/hold staircase nicely
    return float(max(2000.0, 200.0 * fm, 20.0 * fs_samp))


def _sample_message_exact(kind: str, Am: float, fm: float, duration: float, fs_samp: float) -> Tuple[np.ndarray, np.ndarray]:
    if fs_samp <= 0:
        raise ValueError("fs_samp must be positive.")
    Ts = 1.0 / float(fs_samp)

    # exact sampling instants (PAM samples)
    # endpoint excluded to avoid an extra sample exactly at duration
    t_s = np.arange(0.0, float(duration), Ts, dtype=float)
    m_s = gen_message(t_s, kind, Am, fm)
    return t_s, m_s


def _zoh_reconstruct(t: np.ndarray, t_s: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    """
    Zero-order-hold reconstruction on time axis t using sample times t_s and sample values x_s.
    """
    if len(t_s) == 0:
        return np.zeros_like(t, dtype=float)

    idx = np.searchsorted(t_s, t, side="right") - 1
    idx = np.clip(idx, 0, len(x_s) - 1)
    return x_s[idx]


# -------------------------
# PCM: quantize + encode, and decode
# -------------------------

def pcm_quantize(m_s: np.ndarray, n_bits: int, vmin: float, vmax: float) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Uniform quantizer over [vmin, vmax] with L=2^n bins (mid-rise reconstruction).
    Returns (idx, q, delta, L).
    idx in [0..L-1], q are reconstruction values.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    L = int(2 ** int(n_bits))
    x = np.clip(m_s, vmin, vmax)

    delta = float(vmax - vmin) / float(L)
    if delta <= 0:
        raise ValueError("Invalid quantizer range (vmax must be > vmin).")

    idx = np.floor((x - vmin) / delta).astype(int)
    idx = np.clip(idx, 0, L - 1)

    # mid-rise reconstruction value
    q = vmin + (idx + 0.5) * delta
    return idx, q, delta, L


def pcm_encode_from_idx(idx: np.ndarray, n_bits: int) -> Tuple[List[int], List[str]]:
    bitstream: List[int] = []
    codewords: List[str] = []
    for k in idx.tolist():
        cw = format(int(k), f"0{int(n_bits)}b")
        codewords.append(cw)
        bitstream.extend([1 if c == "1" else 0 for c in cw])
    return bitstream, codewords


def pcm_decode_to_idx(bits: List[int], n_bits: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Group bits into n-bit codewords -> idx.
    Drops any remainder bits (and reports them in meta["remainder_bits"]).
    """
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    n_full = len(bits) // n_bits
    rem = len(bits) - n_full * n_bits

    idx_hat = np.zeros(n_full, dtype=int)
    codewords: List[str] = []
    for i in range(n_full):
        chunk = bits[i * n_bits:(i + 1) * n_bits]
        s = "".join("1" if b else "0" for b in chunk)
        codewords.append(s)
        idx_hat[i] = int(s, 2)

    meta = {
        "n_bits": n_bits,
        "n_codewords": n_full,
        "remainder_bits": rem,
        "codewords": codewords,
    }
    return idx_hat, meta


def pcm_reconstruct_from_idx(idx: np.ndarray, n_bits: int, vmin: float, vmax: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Given idx, reconstruct q using the SAME uniform quantizer definition (mid-rise).
    """
    idx = np.asarray(idx, dtype=int)
    L = int(2 ** int(n_bits))
    delta = float(vmax - vmin) / float(L)
    idx2 = np.clip(idx, 0, L - 1)
    q = vmin + (idx2 + 0.5) * delta

    meta = {"L": L, "delta": delta, "vmin": vmin, "vmax": vmax}
    return q, meta


# -------------------------
# DM: encode and decode
# -------------------------

def dm_encode(m_s: np.ndarray, delta: float, *, est0: float) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Delta modulation encoder (book-aligned):

    At each sample k:
      compare x[k] to current staircase estimate est[k]  (comparator)
      output b[k] = 1 if x[k] >= est[k] else 0
      update est[k+1] = est[k] ± delta  (move for the NEXT interval)

    Returns:
      bits, stair_before (est[k]), stair_after (est[k+1] after update stored per k)
    """
    delta = float(delta)
    if delta <= 0:
        raise ValueError("DM delta must be positive.")

    est = float(est0)
    bits: List[int] = []
    stair_before = np.zeros(len(m_s), dtype=float)
    stair_after = np.zeros(len(m_s), dtype=float)

    for i, x in enumerate(m_s.tolist()):
        stair_before[i] = est
        b = 1 if float(x) >= est else 0
        bits.append(b)
        est += delta if b == 1 else -delta
        stair_after[i] = est

    return bits, stair_before, stair_after

def dm_decode(bits: List[int], delta: float, *, est0: float) -> np.ndarray:
    delta = float(delta)
    if delta <= 0:
        raise ValueError("DM delta must be positive.")

    est = float(est0)
    stair = np.zeros(len(bits), dtype=float)

    for i, b in enumerate(bits):
        est += delta if int(b) == 1 else -delta
        stair[i] = est

    return stair


# -------------------------
# Main simulator
# -------------------------

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
    """
    Book-aligned pipeline:

    PCM:
      Analog m(t) -> PAM sampler (fs_samp) -> Quantizer (L=2^n) -> Encoder (n-bit blocks)
      Then (optionally) line-code the produced bitstream for transmission.
      Receiver side (added): line-decode -> PCM decode -> reconstructed samples.

    DM:
      Analog m(t) -> PAM sampler (fs_samp) -> DM (1 bit per sample, step size delta)
      Then line-code, then line-decode, then DM decode staircase.

    We keep step-by-step visualization data in meta.
    """

    technique = str(technique)
    if technique not in ("PCM", "DM"):
        raise ValueError("technique must be PCM or DM")

    # --- Sampling rate for PAM ---
    fm = float(fm)
    fs_mult = int(fs_mult)
    
    if fm <= 0:
        raise ValueError("fm must be positive.")
    if fs_mult <= 0:
        raise ValueError("fs_mult must be positive.")
    
    fs_samp = float(fs_mult) * fm
    Ts = 1.0 / fs_samp

    # exact PAM sampling
    t_s, m_s = _sample_message_exact(kind, Am, fm, duration, fs_samp)

    # --- Display time axis / analog message for plotting ---
    fs_display = _choose_display_fs(float(fm), float(fs_samp))
    N = int(np.ceil(float(duration) * fs_display))
    t = make_time_axis(N, fs_display)
    m = gen_message(t, kind, Am, fm)

    # --- Linecode params (digital waveform sampling) ---
    # In your app, params.fs already equals Ns/Tb; keep that consistent.
    lc_params = SimParams(
        fs=float(params.fs),
        Tb=float(params.Tb),
        samples_per_bit=int(params.samples_per_bit),
        Ac=float(params.Ac),
        fc=float(params.fc),
    )

    meta: Dict[str, Any] = {
        "kind": kind,
        "technique": technique,
        "Am": float(Am),
        "fm": float(fm),
        "duration": float(duration),
        "fs_mult": int(fs_mult),  # <-- add this
        "sampler": {
            "fs_samp": float(fs_samp),
            "Ts": float(Ts),
            "fs_mult": int(fs_mult),
            "num_samples": int(len(m_s)),
        },
        "fs_display": float(fs_display),
    }

    signals: Dict[str, np.ndarray] = {"m(t)": m}
    bits_out: Dict[str, List[int]] = {}

    # ---------------- PCM ----------------
    if technique == "PCM":
        n_bits = int(pcm_nbits)
        vmin, vmax = -float(Am), +float(Am)

        idx, q, q_delta, L = pcm_quantize(m_s, n_bits, vmin=vmin, vmax=vmax)
        bitstream, codewords = pcm_encode_from_idx(idx, n_bits=n_bits)

        # linecode tx and linecode rx (receiver stage)
        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        # PCM receiver decode
        idx_hat, pcm_dec_meta = pcm_decode_to_idx(bitstream_rx, n_bits=n_bits)
        q_hat, pcm_rec_meta = pcm_reconstruct_from_idx(idx_hat, n_bits=n_bits, vmin=vmin, vmax=vmax)

        # reconstruct on display axis
        recon_tx = _zoh_reconstruct(t, t_s, q)
        # For rx, the sample count could differ if bits were truncated; align times
        t_s_rx = t_s[:len(q_hat)] if len(q_hat) <= len(t_s) else np.arange(len(q_hat), dtype=float) * Ts
        recon_rx = _zoh_reconstruct(t, t_s_rx, q_hat)

        signals.update({
            "linecode": line_tx,
            "recon_tx": recon_tx,
            "recon_rx": recon_rx,
        })

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Step-by-step table rows (book Fig 5.17 style)
        steps = []
        for k in range(len(m_s)):
            steps.append({
                "k": k,
                "t_s": float(t_s[k]),
                "PAM sample": float(m_s[k]),
                "Code number": int(idx[k]),
                "PCM code": codewords[k],
                "Quantized q": float(q[k]),
            })

        meta["pcm"] = {
            "n_bits": n_bits,
            "L": int(L),
            "delta": float(q_delta),
            "vmin": float(vmin),
            "vmax": float(vmax),
            "idx": idx,
            "q": q,
            "codewords": codewords,
            "steps": steps,
            # optional SNR estimate from book (uniform quantizer)
            "snr_db_est": float(6.02 * n_bits + 1.76),
        }

        meta["pcm_rx"] = {
            "idx_hat": idx_hat,
            "q_hat": q_hat,
            "decode_meta": pcm_dec_meta,
            "recon_meta": pcm_rec_meta,
        }

        # include a staircase for plotting (sample-wise, not on dense axis)
        meta["stair_tx"] = {"t_s": t_s, "x": q}
        meta["stair_rx"] = {"t_s": t_s_rx, "x": q_hat}

        # linecode stage meta + match
        meta["linecode"] = {
            "scheme": linecode_scheme,
            "encode": lc_enc_meta,
            "decode": lc_dec_meta,
            "match": (bitstream_rx == bitstream),
            "bit_len": int(len(bitstream)),
        }

    # ---------------- DM ----------------
    else:
        delta = float(dm_delta)

        # Book-style default: start staircase at 0 (signals here are centered around 0)
        est0 = 0.0

        bitstream, stair_before, stair_after = dm_encode(m_s, delta=delta, est0=est0)

        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        stair_rx_after = dm_decode(bitstream_rx, delta=delta, est0=est0)

        # Build staircase sequences that include the INITIAL estimate at t=0
        # so the first interval holds est0 (this matches the book’s staircase notion).
        if len(t_s) > 0:
            t_stair = np.concatenate([[t_s[0]], t_s])
        else:
            t_stair = np.array([0.0], dtype=float)

        x_tx_stair = np.concatenate([[est0], stair_after])
        x_rx_stair = np.concatenate([[est0], stair_rx_after])

        # Align RX in case of any truncation/mismatch in bit count
        n_rx = min(len(t_stair), len(x_rx_stair))
        t_rx_stair = t_stair[:n_rx]
        x_rx_stair = x_rx_stair[:n_rx]

        # recon on display axis (ZOH)
        recon_tx = _zoh_reconstruct(t, t_stair, x_tx_stair)
        recon_rx = _zoh_reconstruct(t, t_rx_stair, x_rx_stair)

        signals.update({
            "linecode": line_tx,
            "recon_tx": recon_tx,
            "recon_rx": recon_rx,
        })

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Steps table (comparator + staircase update)
        steps = []
        n_show = min(len(m_s), len(bitstream), len(stair_before), len(stair_after))
        for k in range(n_show):
            steps.append({
                "k": k,
                "t_s": float(t_s[k]),
                "Input sample x[k]": float(m_s[k]),
                "Stair (before)": float(stair_before[k]),
                "DM bit": int(bitstream[k]),
                "Stair (after)": float(stair_after[k]),
            })

        meta["dm"] = {
            "delta": float(delta),
            "est0": float(est0),
            "steps": steps,
        }

        meta["dm_rx"] = {
            "stair_rx_after": stair_rx_after,
        }

        # Use the staircase sequences (with initial point) for the plot
        meta["stair_tx"] = {"t_s": t_stair, "x": x_tx_stair}
        meta["stair_rx"] = {"t_s": t_rx_stair, "x": x_rx_stair}

        meta["linecode"] = {
            "scheme": linecode_scheme,
            "encode": lc_enc_meta,
            "decode": lc_dec_meta,
            "match": (bitstream_rx == bitstream),
            "bit_len": int(len(bitstream)),
        }

    # Common sampled info for UI
    meta["sampled"] = {"t_s": t_s, "m_s": m_s}

    # Time axis for linecode waveform
    t_bits = make_time_axis(len(signals["linecode"]), lc_params.fs)
    meta["t_bits"] = t_bits

    # Provide a compact summary keys (so app can show summary like other modes)
    meta["summary"] = {
        "scheme": f"{technique} + {linecode_scheme}",
        "match": bool(meta["linecode"]["match"]),
        "input_len": int(meta["linecode"]["bit_len"]),
        "fs": float(params.fs),
        "Tb": float(params.Tb),
        "samples_per_bit": int(params.samples_per_bit),
    }

    return SimResult(
        t=t,
        signals=signals,
        bits=bits_out,
        meta=meta,
    )
