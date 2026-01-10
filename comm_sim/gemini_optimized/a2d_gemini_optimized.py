from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from utils import SimParams, SimResult, make_time_axis
# We import d2d line encoding/decoding as required by the pipeline
from d2d import line_encode, line_decode


# -------------------------
# Message generation (analog)
# -------------------------

def gen_message(t: np.ndarray, kind: str, Am: float, fm: float) -> np.ndarray:
    # Optimized: t is already numpy array, operations are vectorized.
    if kind == "sine":
        return Am * np.sin(2 * np.pi * fm * t)

    if kind == "square":
        # Use numpy sin and where for speed
        s = np.sin(2 * np.pi * fm * t)
        return Am * np.where(s >= 0.0, 1.0, -1.0)

    if kind == "triangle":
        # Vectorized sawtooth calc
        x = (t * fm) % 1.0
        # 4 * abs(x - 0.5) - 1.0 generates triangle
        tri = 4 * np.abs(x - 0.5) - 1.0
        return Am * (-tri)

    raise ValueError(f"Unknown message type: {kind}")


def _choose_display_fs(fm: float, fs_samp: float) -> float:
    # Keep logic identical to original
    return float(max(2000.0, 200.0 * fm, 20.0 * fs_samp))


def _sample_message_exact(kind: str, Am: float, fm: float, duration: float, fs_samp: float) -> Tuple[np.ndarray, np.ndarray]:
    if fs_samp <= 0:
        raise ValueError("fs_samp must be positive.")
    Ts = 1.0 / float(fs_samp)

    # exact sampling instants
    t_s = np.arange(0.0, float(duration), Ts, dtype=float)
    m_s = gen_message(t_s, kind, Am, fm)
    return t_s, m_s


def _zoh_reconstruct(t: np.ndarray, t_s: np.ndarray, x_s: np.ndarray) -> np.ndarray:
    """
    Zero-order-hold reconstruction (Vectorized).
    """
    if len(t_s) == 0:
        return np.zeros_like(t, dtype=float)

    # np.searchsorted is highly optimized C-level binary search
    idx = np.searchsorted(t_s, t, side="right") - 1
    # Clip indices to valid range
    idx = np.clip(idx, 0, len(x_s) - 1)
    return x_s[idx]


# -------------------------
# PCM: quantize + encode, and decode
# -------------------------

def pcm_quantize(m_s: np.ndarray, n_bits: int, vmin: float, vmax: float) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Uniform quantizer (Vectorized).
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    L = int(2 ** int(n_bits))
    
    # Vectorized clip
    x = np.clip(m_s, vmin, vmax)

    delta = float(vmax - vmin) / float(L)
    if delta <= 0:
        raise ValueError("Invalid quantizer range (vmax must be > vmin).")

    # Vectorized floor and cast
    idx = np.floor((x - vmin) / delta).astype(int)
    idx = np.clip(idx, 0, L - 1)

    # Vectorized reconstruction
    q = vmin + (idx + 0.5) * delta
    return idx, q, delta, L


def pcm_encode_from_idx(idx: np.ndarray, n_bits: int) -> Tuple[List[int], List[str]]:
    """
    Vectorized bit encoding.
    Replaces the loop-based string formatting for bit generation.
    """
    # 1. Generate Bitstream (Vectorized)
    # Create shift array: [n-1, n-2, ... 0]
    shifts = np.arange(n_bits - 1, -1, -1, dtype=int)
    
    # Broadcast: (N, 1) >> (n_bits,) -> (N, n_bits)
    # idx must be integer type
    idx_int = idx.astype(int)
    
    # Extract bits: shift right and mask with 1
    bits_matrix = (idx_int[:, None] >> shifts) & 1
    
    # Flatten to get serial bitstream
    bitstream = bits_matrix.flatten().tolist()

    # 2. Generate Codewords (List comprehension)
    # String formatting is still needed for metadata/UI, but list comp is faster than append loop
    # We use the f-string in a comprehension
    fmt = f"0{int(n_bits)}b"
    codewords = [format(val, fmt) for val in idx_int]
    
    return bitstream, codewords


def pcm_decode_to_idx(bits: List[int], n_bits: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Vectorized bit decoding.
    """
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    n_full = len(bits) // n_bits
    rem = len(bits) - n_full * n_bits

    if n_full == 0:
        return np.array([], dtype=int), {
            "n_bits": n_bits, "n_codewords": 0, "remainder_bits": rem, "codewords": []
        }

    # Convert to array
    bits_arr = np.array(bits, dtype=int)
    
    # Truncate to multiple of n_bits
    valid_len = n_full * n_bits
    bits_matrix = bits_arr[:valid_len].reshape(n_full, n_bits)
    
    # Vectorized binary to int: Dot product with powers of 2
    # Powers: [2^(n-1), ..., 1]
    powers = 1 << np.arange(n_bits - 1, -1, -1, dtype=int)
    idx_hat = bits_matrix.dot(powers)
    
    # Reconstruct codewords list for meta (optimized list comp)
    # We join bits back to strings. 
    # Since we have the integer values (idx_hat), it's actually faster to format the integers
    # than to slice strings from the bit list again.
    fmt = f"0{n_bits}b"
    codewords = [format(val, fmt) for val in idx_hat]

    meta = {
        "n_bits": n_bits,
        "n_codewords": n_full,
        "remainder_bits": rem,
        "codewords": codewords,
    }
    return idx_hat, meta


def pcm_reconstruct_from_idx(idx: np.ndarray, n_bits: int, vmin: float, vmax: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Vectorized reconstruction.
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
    Delta modulation encoder.
    Optimization: Use pre-allocated numpy arrays and local variable caching to speed up the loop.
    Note: DM encoding is inherently sequential (feedback), so we cannot fully vectorize,
    but we can minimize Python object overhead.
    """
    delta = float(delta)
    if delta <= 0:
        raise ValueError("DM delta must be positive.")

    n = len(m_s)
    est = float(est0)
    
    # Pre-allocate output arrays
    stair_before = np.empty(n, dtype=float)
    stair_after = np.empty(n, dtype=float)
    bits_arr = np.empty(n, dtype=int)
    
    # Convert input to numpy array for fast indexing
    # (m_s should already be one, but ensure)
    m_s_arr = np.asarray(m_s, dtype=float)
    
    # Optimized loop
    for i in range(n):
        stair_before[i] = est
        # Compare
        if m_s_arr[i] >= est:
            bits_arr[i] = 1
            est += delta
        else:
            bits_arr[i] = 0
            est -= delta
        stair_after[i] = est

    return bits_arr.tolist(), stair_before, stair_after


def dm_decode(bits: List[int], delta: float, *, est0: float) -> np.ndarray:
    """
    Delta modulation decoder.
    Optimization: Fully vectorized using cumsum.
    """
    delta = float(delta)
    if delta <= 0:
        raise ValueError("DM delta must be positive.")

    bits_arr = np.array(bits, dtype=int)
    
    # Map bits: 1 -> +delta, 0 -> -delta
    # 1 => 1, 0 => -1.   (b*2 - 1) gives map: 1->1, 0->-1
    steps = (bits_arr * 2 - 1) * delta
    
    # Cumulative sum
    stair = float(est0) + np.cumsum(steps)
    
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
    Optimized main simulator. 
    Maintains exact API and metadata structure of the original.
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

    # --- Display time axis ---
    fs_display = _choose_display_fs(float(fm), float(fs_samp))
    N = int(np.ceil(float(duration) * fs_display))
    t = make_time_axis(N, fs_display)
    m = gen_message(t, kind, Am, fm)

    # --- Linecode params ---
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

        # 1. Quantize (Vectorized)
        idx, q, q_delta, L = pcm_quantize(m_s, n_bits, vmin=vmin, vmax=vmax)
        
        # 2. Encode (Vectorized bit packing)
        bitstream, codewords = pcm_encode_from_idx(idx, n_bits=n_bits)

        # 3. Line Coding (External D2D module)
        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        # 4. PCM Decode (Vectorized)
        idx_hat, pcm_dec_meta = pcm_decode_to_idx(bitstream_rx, n_bits=n_bits)
        q_hat, pcm_rec_meta = pcm_reconstruct_from_idx(idx_hat, n_bits=n_bits, vmin=vmin, vmax=vmax)

        # 5. Reconstruction (Vectorized ZOH)
        recon_tx = _zoh_reconstruct(t, t_s, q)
        t_s_rx = t_s[:len(q_hat)] if len(q_hat) <= len(t_s) else np.arange(len(q_hat), dtype=float) * Ts
        recon_rx = _zoh_reconstruct(t, t_s_rx, q_hat)

        signals.update({
            "linecode": line_tx,
            "recon_tx": recon_tx,
            "recon_rx": recon_rx,
        })

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Metadata: "steps" table (List comp for speed, required by tests)
        # Using zip for iteration is faster than range indexing
        steps = [
            {
                "k": k,
                "t_s": float(tk),
                "PAM sample": float(mk),
                "Code number": int(ik),
                "PCM code": ck,
                "Quantized q": float(qk),
            }
            for k, (tk, mk, ik, ck, qk) in enumerate(zip(t_s, m_s, idx, codewords, q))
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
        est0 = 0.0

        # 1. DM Encode (Semi-vectorized/Pre-allocated)
        bitstream, stair_before, stair_after = dm_encode(m_s, delta=delta, est0=est0)

        # 2. Line Code
        line_tx, lc_enc_meta = line_encode(bitstream, linecode_scheme, lc_params)
        bitstream_rx, lc_dec_meta = line_decode(line_tx, linecode_scheme, lc_params)

        # 3. DM Decode (Vectorized)
        stair_rx_after = dm_decode(bitstream_rx, delta=delta, est0=est0)

        # Staircase alignment for plotting
        if len(t_s) > 0:
            t_stair = np.concatenate([[t_s[0]], t_s])
        else:
            t_stair = np.array([0.0], dtype=float)

        x_tx_stair = np.concatenate([[est0], stair_after])
        x_rx_stair = np.concatenate([[est0], stair_rx_after])

        n_rx = min(len(t_stair), len(x_rx_stair))
        t_rx_stair = t_stair[:n_rx]
        x_rx_stair = x_rx_stair[:n_rx]

        # 4. Reconstruction (Vectorized)
        recon_tx = _zoh_reconstruct(t, t_stair, x_tx_stair)
        recon_rx = _zoh_reconstruct(t, t_rx_stair, x_rx_stair)

        signals.update({
            "linecode": line_tx,
            "recon_tx": recon_tx,
            "recon_rx": recon_rx,
        })

        bits_out["bitstream"] = bitstream
        bits_out["decoded_bitstream"] = bitstream_rx

        # Metadata: "steps" table
        n_show = min(len(m_s), len(bitstream), len(stair_before), len(stair_after))
        steps = [
            {
                "k": k,
                "t_s": float(t_s[k]),
                "Input sample x[k]": float(m_s[k]),
                "Stair (before)": float(stair_before[k]),
                "DM bit": int(bitstream[k]),
                "Stair (after)": float(stair_after[k]),
            }
            # Use islice/range logic or simple slicing for safe iteration
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