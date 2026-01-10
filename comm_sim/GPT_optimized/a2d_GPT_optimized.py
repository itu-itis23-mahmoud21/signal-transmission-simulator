from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from utils import SimParams, SimResult, make_time_axis


# ==========================================================
# Line coder / decoder import
# ==========================================================
# Prefer the GPT-optimized d2d implementation if it's present and defines the API.
# Fall back to the original d2d otherwise.
def _resolve_linecodec():
    for modname in ("d2d_GPT_optmizied", "d2d_GPT_optimized", "d2d"):
        try:
            mod = __import__(modname, fromlist=["line_encode", "line_decode"])
            le = getattr(mod, "line_encode", None)
            ld = getattr(mod, "line_decode", None)
            if callable(le) and callable(ld):
                return le, ld
        except Exception:
            continue
    raise ImportError("Could not import a working line_encode/line_decode from d2d modules.")


line_encode, line_decode = _resolve_linecodec()

# -------------------------
# Signal generators
# -------------------------
def gen_message(t: np.ndarray, kind: str, Am: float, fm: float) -> np.ndarray:
    kind = str(kind).lower()
    Am = float(Am)
    fm = float(fm)

    if t.size == 0:
        return np.array([], dtype=float)

    if kind == "sine":
        return Am * np.sin(2 * np.pi * fm * t)

    if kind == "square":
        return Am * np.where(np.sin(2 * np.pi * fm * t) >= 0.0, 1.0, -1.0)

    if kind == "triangle":
        x = np.mod(t * fm, 1.0)  # [0,1)
        tri = 4.0 * np.abs(x - 0.5) - 1.0  # [-1,1]
        return Am * (-tri)

    raise ValueError(f"Unknown message kind: {kind}")


def _choose_display_fs(fm: float, fs_samp: float) -> float:
    fm = float(fm)
    fs_samp = float(fs_samp)
    return max(2000.0, 2000.0 * fm, 20.0 * fs_samp)


def _sample_message_exact(kind: str, Am: float, fm: float, duration: float, fs_samp: float) -> Tuple[np.ndarray, np.ndarray]:
    fs_samp = float(fs_samp)
    if fs_samp <= 0:
        raise ValueError("fs_samp must be positive.")
    Ts = 1.0 / fs_samp
    t_s = np.arange(0.0, float(duration), Ts, dtype=float)
    m_s = gen_message(t_s, kind, Am, fm)
    return t_s, m_s


def _zoh_reconstruct(t: np.ndarray, t_s: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.array([], dtype=float)
    if t_s.size == 0 or x_s.size == 0:
        return np.zeros_like(t, dtype=float)

    idx = np.searchsorted(t_s, t, side="right") - 1
    idx = np.clip(idx, 0, len(x_s) - 1)
    return np.asarray(x_s, dtype=float)[idx]


# -------------------------
# PCM helpers (optimized bit packing/unpacking)
# -------------------------
def pcm_quantize(m_s: np.ndarray, n_bits: int, *, vmin: float, vmax: float):
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        raise ValueError("vmax must be > vmin.")

    L = 1 << n_bits
    delta = (vmax - vmin) / float(L)

    if m_s.size == 0:
        idx = np.array([], dtype=int)
        q = np.array([], dtype=float)
        return idx, q, float(delta), int(L)

    x = (np.asarray(m_s, dtype=float) - vmin) / delta
    idx = np.floor(x).astype(int)
    np.clip(idx, 0, L - 1, out=idx)

    q = vmin + (idx.astype(float) + 0.5) * delta
    return idx, q, float(delta), int(L)


def _idx_to_bitstream(idx: np.ndarray, n_bits: int) -> List[int]:
    if idx.size == 0:
        return []
    idx_u = np.asarray(idx, dtype=np.uint32).reshape(-1, 1)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint32).reshape(1, -1)
    bits_mat = ((idx_u >> shifts) & 1).astype(np.uint8)
    return bits_mat.reshape(-1).tolist()


def pcm_encode_from_idx(idx: np.ndarray, *, n_bits: int) -> Tuple[List[int], List[str]]:
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    idx_arr = np.asarray(idx, dtype=int)
    if idx_arr.size == 0:
        return [], []

    # Strings are needed for UI/tests; keep them, but pack bits via vectorization.
    codewords = [format(int(k), f"0{n_bits}b") for k in idx_arr.tolist()]
    bitstream = _idx_to_bitstream(idx_arr, n_bits)
    return bitstream, codewords


def pcm_decode_to_idx(bits: List[int], *, n_bits: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n_full = int(b.size // n_bits)
    rem = int(b.size - n_full * n_bits)

    if n_full <= 0:
        meta = {
            "n_bits": int(n_bits),
            "n_codewords": 0,
            "remainder_bits": int(rem),
            "codewords": [],
        }
        return np.array([], dtype=int), meta

    b_full = b[: n_full * n_bits].reshape(n_full, n_bits).astype(np.uint64)

    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    weights = (np.uint64(1) << shifts).reshape(1, -1)
    idx_hat = (b_full * weights).sum(axis=1).astype(int)

    codewords = [format(int(k), f"0{n_bits}b") for k in idx_hat.tolist()]

    meta = {
        "n_bits": int(n_bits),
        "n_codewords": int(n_full),
        "remainder_bits": int(rem),
        "codewords": codewords,
    }
    return idx_hat, meta


def pcm_reconstruct_from_idx(idx: np.ndarray, *, n_bits: int, vmin: float, vmax: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    vmin = float(vmin)
    vmax = float(vmax)
    L = 1 << n_bits
    delta = (vmax - vmin) / float(L)

    idx_arr = np.asarray(idx, dtype=float)
    q_hat = vmin + (idx_arr + 0.5) * delta

    meta = {"delta": float(delta), "L": int(L), "vmin": float(vmin), "vmax": float(vmax)}
    return np.asarray(q_hat, dtype=float), meta


# -------------------------
# DM helpers
# -------------------------
def dm_encode(m_s: np.ndarray, *, delta: float, est0: float = 0.0):
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be positive.")
    est = float(est0)

    x = np.asarray(m_s, dtype=float)
    n = int(x.size)
    if n == 0:
        return [], np.array([], dtype=float), np.array([], dtype=float)

    bits = [0] * n
    stair_before = np.empty(n, dtype=float)
    stair_after = np.empty(n, dtype=float)

    # Iterating over list of floats is usually faster than numpy scalar iteration.
    for i, xi in enumerate(x.tolist()):
        stair_before[i] = est
        b = 1 if xi >= est else 0
        bits[i] = b
        est = est + delta if b == 1 else est - delta
        stair_after[i] = est

    return bits, stair_before, stair_after


def dm_decode(bits: List[int], *, delta: float, est0: float = 0.0) -> np.ndarray:
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be positive.")

    if not bits:
        return np.array([], dtype=float)

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
      Then line-code the produced bitstream for transmission.
      Receiver: line-decode -> PCM decode -> reconstructed samples.

    DM:
      Analog m(t) -> PAM sampler (fs_samp) -> DM (1 bit per sample, step size delta)
      Then line-code, then line-decode, then DM decode staircase.

    Step-by-step visualization data is kept in meta (for UI/tests).
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
        "fs_mult": int(fs_mult),
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

        # linecode tx and rx
        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        # PCM receiver decode
        idx_hat, pcm_dec_meta = pcm_decode_to_idx(bitstream_rx, n_bits=n_bits)
        q_hat, pcm_rec_meta = pcm_reconstruct_from_idx(idx_hat, n_bits=n_bits, vmin=vmin, vmax=vmax)

        # reconstruct on display axis
        recon_tx = _zoh_reconstruct(t, t_s, q)
        # For rx, align times if truncated
        t_s_rx = t_s[: len(q_hat)] if len(q_hat) <= len(t_s) else np.arange(len(q_hat), dtype=float) * Ts
        recon_rx = _zoh_reconstruct(t, t_s_rx, q_hat)

        signals.update(
            {
                "linecode": line_tx,
                "recon_tx": recon_tx,
                "recon_rx": recon_rx,
            }
        )

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Step-by-step table rows (book Fig 5.17 style)
        steps: List[Dict[str, Any]] = []
        if len(m_s) > 0:
            t_s_list = t_s.tolist()
            m_s_list = np.asarray(m_s, dtype=float).tolist()
            idx_list = np.asarray(idx, dtype=int).tolist()
            q_list = np.asarray(q, dtype=float).tolist()

            steps = [
                {
                    "k": k,
                    "t_s": float(ts),
                    "PAM sample": float(ms),
                    "Code number": int(ii),
                    "PCM code": cw,
                    "Quantized q": float(qq),
                }
                for k, (ts, ms, ii, cw, qq) in enumerate(zip(t_s_list, m_s_list, idx_list, codewords, q_list))
            ]

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

        # Default: start staircase at 0 (signals are centered around 0)
        est0 = 0.0

        bitstream, stair_before, stair_after = dm_encode(m_s, delta=delta, est0=est0)

        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        stair_rx_after = dm_decode(bitstream_rx, delta=delta, est0=est0)

        # Build staircase sequences that include the INITIAL estimate at t=0
        if len(t_s) > 0:
            t_stair = np.concatenate([[t_s[0]], t_s])
        else:
            t_stair = np.array([0.0], dtype=float)

        x_tx_stair = np.concatenate([[est0], stair_after])
        x_rx_stair = np.concatenate([[est0], stair_rx_after])

        # Align RX in case of any truncation/mismatch
        n_rx = min(len(t_stair), len(x_rx_stair))
        t_rx_stair = t_stair[:n_rx]
        x_rx_stair = x_rx_stair[:n_rx]

        recon_tx = _zoh_reconstruct(t, t_stair, x_tx_stair)
        recon_rx = _zoh_reconstruct(t, t_rx_stair, x_rx_stair)

        signals.update(
            {
                "linecode": line_tx,
                "recon_tx": recon_tx,
                "recon_rx": recon_rx,
            }
        )

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Steps table (comparator + staircase update)
        steps: List[Dict[str, Any]] = []
        n_show = min(len(m_s), len(bitstream), len(stair_before), len(stair_after))
        if n_show > 0:
            t_s_list = t_s.tolist()
            m_s_list = np.asarray(m_s, dtype=float).tolist()
            sb_list = np.asarray(stair_before, dtype=float).tolist()
            sa_list = np.asarray(stair_after, dtype=float).tolist()
            bits_list = [int(b) for b in bitstream]

            steps = [
                {
                    "k": k,
                    "t_s": float(t_s_list[k]),
                    "Input sample x[k]": float(m_s_list[k]),
                    "Stair (before)": float(sb_list[k]),
                    "DM bit": int(bits_list[k]),
                    "Stair (after)": float(sa_list[k]),
                }
                for k in range(n_show)
            ]

        meta["dm"] = {
            "delta": float(delta),
            "est0": float(est0),
            "steps": steps,
        }

        meta["dm_rx"] = {
            "stair_rx_after": stair_rx_after,
        }

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

    # Summary (matches other modes' style)
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
