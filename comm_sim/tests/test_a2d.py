# test_a2d.py
#
# Analog → Digital (A2D) digitization unit tests for `simulate_a2d(...)`.
#
# What this test suite verifies
# -----------------------------
# 1) End-to-end correctness under ideal conditions (core requirement)
#    - For both digitization techniques (PCM, DM) and all supported message waveforms,
#      the produced bitstream must survive line coding + line decoding losslessly:
#        decoded_bitstream == bitstream
#      and `meta["linecode"]["match"]` must be True.
#
# 2) Technique coverage + waveform coverage
#    - Techniques: PCM (Pulse Code Modulation), DM (Delta Modulation)
#    - Waveforms: sine, square, triangle
#    - Line codes (as used in the UI): NRZ-L, NRZI, Manchester, Bipolar-AMI
#
# 3) PCM-specific invariants (beyond match)
#    - Uniform mid-rise quantizer: L=2^n, Δ=(vmax-vmin)/L, q=vmin+(idx+0.5)Δ
#    - idx is clipped to [0, L-1]
#    - bitstream length = num_samples * n_bits
#    - codewords are exactly n_bits long and match idx values
#    - RX reconstruction equals TX reconstruction in an ideal line-coded channel
#
# 4) DM-specific invariants (book-aligned comparator + staircase update)
#    - bit[k] = 1 iff x[k] >= stair_before[k]
#    - stair_after[k] = stair_before[k] ± Δ (sign depends on bit)
#    - bitstream length = num_samples (1 bit per sample)
#    - RX staircase/reconstruction equals TX staircase/reconstruction when linecode match holds
#
# 5) Line-code waveform invariants (sanity)
#    - Binary line codes use only {+1, -1}
#    - AMI uses {+1, 0, -1}
#    - Manchester uses even Ns internally; odd Ns must be auto-adjusted to even
#
# 6) Edge cases and invalid-parameter behavior
#    - Unknown message type raises
#    - Non-positive fm or fs_mult raises (fs_samp must be positive)
#    - Invalid PCM n_bits raises
#    - Invalid DM delta raises
#    - Unknown linecode scheme raises
#    - duration=0 produces empty signals/bitstreams but remains consistent
#
# 7) Long-run stress tests
#    - Exercises “realistic worst-case” parameter combinations (high fm, high fs_mult,
#      longer duration, larger Ns) to catch numerical issues, performance regressions,
#      metadata shape mismatches, and any hidden state assumptions.
#    - Runs both techniques:
#        * PCM: tests across all line codes, both low and high n_bits regimes, and across
#          all message waveforms (sine/square/triangle).
#        * DM: tests across all line codes with both very small Δ (slope-overload-like
#          behavior) and large Δ, plus an extra hard case (square wave + Manchester + odd Ns).
#    - Verifies in each stress case:
#        * Ideal-channel roundtrip remains lossless: decoded_bitstream == bitstream and match=True
#        * Core arrays remain finite (no NaN/Inf) for m(t), linecode, recon_tx, recon_rx
#        * Key size/shape relationships hold (bit_len·Ns == len(linecode), etc.)
#
# How to run
# ----------
# Set the environment variable `TEST_TARGET` to either `original` or `optimized` to select
# which implementation to test. If not set, defaults to `original`.
#  - Example (Linux/Mac):
#      TEST_TARGET=optimized pytest -v comm_sim/tests/test_a2d.py
#  - Example (Windows CMD):
#      set TEST_TARGET=optimized&& pytest -v comm_sim/tests/test_a2d.py


from __future__ import annotations

import os
import sys
import random
from typing import Any, Dict, List

import numpy as np
import pytest

from utils import SimParams


# ==========================================
# Dynamic Import Logic (Environment Switch)
# ==========================================
test_target = os.getenv("TEST_TARGET", "original")

if test_target == "optimized":
    # 1. Resolve path relative to THIS test file
    current_test_dir = os.path.dirname(os.path.abspath(__file__))     # .../comm_sim/tests
    project_root = os.path.dirname(current_test_dir)                  # .../comm_sim
    optimized_folder = os.path.join(project_root, "gemini_optimized") # .../comm_sim/gemini_optimized

    print(f"\n>>> TARGET MODE: Optimized")

    # 2. Add to system path
    if optimized_folder not in sys.path:
        sys.path.insert(0, optimized_folder)

    # 3. Import with fallback for the typo
    try:
        from a2d_gemini_optimized import simulate_a2d
    except ImportError:
        try:
            # Fallback for the file on your disk
            from a2d_gemini_optmizied import simulate_a2d
            print(">>> NOTE: Imported 'a2d_gemini_optmizied.py' (detected filename typo on disk)")
        except ImportError as e:
            sys.exit(f"CRITICAL ERROR: Could not import optimized file.\nChecked path: {optimized_folder}\nError: {e}")

else:
    print("\n>>> TARGET MODE: Original (a2d.py) <<<\n")
    try:
        from a2d import simulate_a2d
    except ImportError:
         sys.exit("CRITICAL ERROR: Could not import 'a2d.py'. Ensure 'comm_sim' is in your Python path.")

# DEBUG: Verify exactly which file is loaded
print(f"DEBUG: simulate_a2d is loaded from: {simulate_a2d.__code__.co_filename}\n")


# -------------------------
# Shared utilities / helpers
# -------------------------

TECHNIQUES = ["PCM", "DM"]
KINDS = ["sine", "square", "triangle"]

# Line codes exposed in the A2D UI (we still rely on d2d's implementation under the hood)
LINECODES = ["NRZ-L", "NRZI", "Manchester", "Bipolar-AMI"]


def make_params(Ns: int, Tb: float = 1.0, *, fc: float = 10.0, Ac: float = 1.0) -> SimParams:
    # fs is derived so that "Ns samples per bit" means fs = Ns/Tb.
    return SimParams(
        fs=float(Ns) / float(Tb),
        Tb=float(Tb),
        samples_per_bit=int(Ns),
        Ac=float(Ac),
        fc=float(fc),
    )


def run(kind: str, technique: str, params: SimParams, **kwargs):
    return simulate_a2d(kind, technique, params, **kwargs)


def assert_linecode_match(res, ctx: str = ""):
    lc = res.meta.get("linecode", {}) or {}
    assert bool(lc.get("match", False)), ctx or "Linecode decode mismatch."


def ns_used(res, params: SimParams) -> int:
    # line_encode adjusts Ns to even for Manchester-family; it reports it in encode meta.
    lc = res.meta.get("linecode", {}) or {}
    enc = lc.get("encode", {}) or {}
    return int(enc.get("Ns_adjusted_even", params.samples_per_bit))


def assert_levels_subset(wave: np.ndarray, allowed: set, tol: float = 1e-9):
    if wave.size == 0:
        return
    uniq = set(np.unique(np.round(wave.astype(float), 12)))
    for v in uniq:
        assert any(abs(float(v) - float(a)) <= tol for a in allowed), f"Unexpected level {v} not in {allowed}"


# =====================================
# 0) Basic API / invalid input behavior
# =====================================

def test_unknown_message_kind_raises():
    params = make_params(20)
    with pytest.raises(ValueError):
        run(
            "not-a-waveform",
            "PCM",
            params,
            Am=1.0, fm=5.0, duration=1.0,
            fs_mult=8, pcm_nbits=4,
            linecode_scheme="NRZ-L",
        )


@pytest.mark.parametrize("fm,fs_mult", [(0.0, 8), (-5.0, 8), (5.0, 0), (5.0, -2)])
def test_nonpositive_sampling_rate_inputs_raise(fm, fs_mult):
    params = make_params(20)
    with pytest.raises(ValueError):
        run(
            "sine",
            "PCM",
            params,
            Am=1.0, fm=float(fm), duration=1.0,
            fs_mult=int(fs_mult), pcm_nbits=4,
            linecode_scheme="NRZ-L",
        )


@pytest.mark.parametrize("n_bits", [0, -1, -7])
def test_pcm_invalid_n_bits_raises(n_bits):
    params = make_params(20)
    with pytest.raises(ValueError):
        run(
            "sine",
            "PCM",
            params,
            Am=1.0, fm=5.0, duration=1.0,
            fs_mult=8, pcm_nbits=int(n_bits),
            linecode_scheme="NRZ-L",
        )


@pytest.mark.parametrize("delta", [0.0, -0.1, -2.0])
def test_dm_invalid_delta_raises(delta):
    params = make_params(20)
    with pytest.raises(ValueError):
        run(
            "sine",
            "DM",
            params,
            Am=1.0, fm=5.0, duration=1.0,
            fs_mult=8, dm_delta=float(delta),
            linecode_scheme="NRZ-L",
        )


def test_unknown_linecode_scheme_raises():
    params = make_params(20)
    with pytest.raises(ValueError):
        run(
            "sine",
            "PCM",
            params,
            Am=1.0, fm=5.0, duration=1.0,
            fs_mult=8, pcm_nbits=4,
            linecode_scheme="NOT-A-LINECODE",
        )


def test_samples_per_bit_zero_raises_from_linecode_path():
    # d2d line coding relies on Ns>0; Ns=0 should raise somewhere in encode/decode sampling.
    params = make_params(1)
    params.samples_per_bit = 0
    with pytest.raises(Exception):
        run(
            "sine",
            "PCM",
            params,
            Am=1.0, fm=5.0, duration=1.0,
            fs_mult=8, pcm_nbits=4,
            linecode_scheme="NRZ-L",
        )


# ==========================
# 1) Meta / structure sanity
# ==========================

@pytest.mark.parametrize("technique", TECHNIQUES)
def test_meta_has_expected_top_fields(technique):
    params = make_params(20)
    res = run(
        "sine",
        technique,
        params,
        Am=1.0, fm=5.0, duration=1.0,
        fs_mult=8, pcm_nbits=4, dm_delta=0.1,
        linecode_scheme="NRZ-L",
    )

    meta = res.meta
    for k in ["kind", "technique", "Am", "fm", "duration", "fs_mult", "sampler", "sampled", "linecode", "t_bits", "summary"]:
        assert k in meta

    assert isinstance(meta["sampler"], dict)
    assert isinstance(meta["sampled"], dict)
    assert isinstance(meta["linecode"], dict)
    assert isinstance(meta["summary"], dict)

    # Basic signal/bits presence
    assert "m(t)" in res.signals
    assert "linecode" in res.signals
    assert "bitstream" in res.bits
    assert "decoded_bitstream" in res.bits

@pytest.mark.parametrize("technique", TECHNIQUES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fs_mult", [2, 8, 16])
def test_sampler_meta_is_self_consistent(technique, kind, fs_mult):
    params = make_params(20)
    fm = 5.0
    duration = 1.0

    res = run(
        kind, technique, params,
        Am=1.0, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=4, dm_delta=0.1,
        linecode_scheme="NRZ-L",
    )

    meta = res.meta
    sampler = meta["sampler"]
    sampled = meta["sampled"]

    assert int(meta["fs_mult"]) == int(fs_mult)
    assert int(sampler["fs_mult"]) == int(fs_mult)

    fs_samp = float(sampler["fs_samp"])
    Ts = float(sampler["Ts"])

    assert fs_samp == pytest.approx(float(fs_mult) * float(fm))
    assert Ts == pytest.approx(1.0 / fs_samp)

    t_s = np.asarray(sampled["t_s"], dtype=float)
    m_s = np.asarray(sampled["m_s"], dtype=float)

    assert len(t_s) == len(m_s) == int(sampler["num_samples"])
    if len(t_s) > 0:
        assert t_s[0] == pytest.approx(0.0)
        assert t_s[-1] < duration + 1e-12  # endpoint excluded by design
        if len(t_s) > 1:
            assert np.allclose(np.diff(t_s), Ts, atol=1e-12, rtol=0.0)

# ===========================================
# 2) Roundtrip grid: all techniques and kinds
# ===========================================

@pytest.mark.parametrize("Ns", [20, 21, 60])  # include odd Ns to force Manchester even-adjustment path
@pytest.mark.parametrize("linecode", LINECODES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("technique", TECHNIQUES)
def test_linecode_roundtrip_all_variants(Ns, linecode, kind, technique):
    params = make_params(Ns)
    res = run(
        kind,
        technique,
        params,
        Am=1.0, fm=5.0, duration=1.0,
        fs_mult=8, pcm_nbits=4, dm_delta=0.1,
        linecode_scheme=linecode,
    )

    # Core: produced bitstream survives line code + decode
    assert_linecode_match(res, f"{technique}/{kind}/{linecode} mismatch Ns={Ns}")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]

    # Waveform length must match bit_len * Ns_used (Manchester may adjust Ns)
    bit_len = int(res.meta["linecode"]["bit_len"])
    Ns_used = ns_used(res, params)
    assert len(res.signals["linecode"]) == bit_len * Ns_used
    assert len(res.meta["t_bits"]) == len(res.signals["linecode"])

    # Bitstream length vs sample count per technique
    num_samples = int(res.meta["sampler"]["num_samples"])
    if technique == "PCM":
        assert bit_len == num_samples * 4
    else:
        assert bit_len == num_samples

@pytest.mark.parametrize("linecode", LINECODES)
def test_t_bits_axis_matches_linecode_and_has_correct_spacing(linecode):
    params = make_params(20)
    res = run(
        "sine",
        "PCM",
        params,
        Am=1.0, fm=5.0, duration=1.0,
        fs_mult=8, pcm_nbits=4,
        linecode_scheme=linecode,
    )
    assert_linecode_match(res)

    t_bits = np.asarray(res.meta["t_bits"], dtype=float)
    wave = np.asarray(res.signals["linecode"], dtype=float)

    assert len(t_bits) == len(wave)
    if len(t_bits) > 1:
        dt = np.diff(t_bits)
        assert np.allclose(dt, 1.0 / float(params.fs), atol=1e-12, rtol=0.0)


# ==================================
# 3) Linecode waveform level invariants
# ==================================

@pytest.mark.parametrize("Ns", [20, 21])
@pytest.mark.parametrize("linecode", LINECODES)
def test_linecode_wave_levels_subset(Ns, linecode):
    params = make_params(Ns)
    res = run(
        "sine",
        "PCM",
        params,
        Am=1.0, fm=4.0, duration=1.0,
        fs_mult=8, pcm_nbits=3,
        linecode_scheme=linecode,
    )
    assert_linecode_match(res)

    wave = res.signals["linecode"]
    if linecode in ("NRZ-L", "NRZI", "Manchester"):
        assert_levels_subset(wave, allowed={-1.0, +1.0})
    else:
        # Bipolar-AMI
        assert_levels_subset(wave, allowed={-1.0, 0.0, +1.0})

    # Manchester must use even Ns internally
    if linecode == "Manchester":
        assert ns_used(res, params) % 2 == 0

@pytest.mark.parametrize("Ns_in, Ns_expected", [(20, 20), (21, 22)])
def test_manchester_odd_Ns_rounds_up_by_one(Ns_in, Ns_expected):
    params = make_params(Ns_in)
    res = run(
        "sine",
        "PCM",
        params,
        Am=1.0, fm=5.0, duration=1.0,
        fs_mult=8, pcm_nbits=3,
        linecode_scheme="Manchester",
    )
    assert_linecode_match(res)

    bit_len = int(res.meta["linecode"]["bit_len"])
    wave_len = len(res.signals["linecode"])
    assert bit_len > 0
    assert wave_len % bit_len == 0

    Ns_actual = wave_len // bit_len
    assert Ns_actual == Ns_expected


# ==================================
# 4) PCM: quantizer + codec invariants
# ==================================

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_bits", [2, 3, 4, 6])
def test_pcm_quantizer_invariants(kind, n_bits):
    params = make_params(60)
    Am = 2.0
    fm = 5.0
    fs_mult = 8
    duration = 1.0

    res = run(
        kind,
        "PCM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=n_bits,
        linecode_scheme="NRZ-L",
    )
    assert_linecode_match(res)

    pcm = res.meta.get("pcm", {})
    assert isinstance(pcm, dict)

    L = int(pcm["L"])
    assert L == 2 ** int(n_bits)

    # Delta should match range / L
    delta = float(pcm["delta"])
    assert np.isfinite(delta)
    assert abs(delta - (2.0 * Am) / L) < 1e-9

    idx = np.asarray(pcm["idx"], dtype=int)
    q = np.asarray(pcm["q"], dtype=float)
    vmin = float(pcm["vmin"])
    vmax = float(pcm["vmax"])

    assert np.all(idx >= 0) and np.all(idx <= (L - 1))
    assert q.shape == idx.shape

    # Mid-rise reconstruction rule: q = vmin + (idx + 0.5)*Δ
    q_ref = vmin + (idx.astype(float) + 0.5) * delta
    assert np.allclose(q, q_ref, atol=1e-12)

    # q must sit inside the open interval (vmin, vmax) for mid-rise (except numerical edges)
    assert np.all(q >= (vmin + 0.5 * delta - 1e-12))
    assert np.all(q <= (vmax - 0.5 * delta + 1e-12))

    # Codewords sanity
    codewords = pcm["codewords"]
    assert isinstance(codewords, list)
    assert len(codewords) == len(idx)
    for cw, i in zip(codewords, idx.tolist()):
        assert isinstance(cw, str)
        assert len(cw) == n_bits
        assert set(cw).issubset({"0", "1"})
        assert int(cw, 2) == int(i)

    # Bitstream length is samples * n_bits
    num_samples = int(res.meta["sampler"]["num_samples"])
    assert len(res.bits["bitstream"]) == num_samples * n_bits
    assert len(res.bits["decoded_bitstream"]) == num_samples * n_bits

    # TX/RX staircases should match exactly in an ideal line-coded channel
    stair_tx = res.meta["stair_tx"]["x"]
    stair_rx = res.meta["stair_rx"]["x"]
    assert np.allclose(stair_tx, stair_rx, atol=1e-12)

    # Reconstructions on dense axis should match too
    assert np.allclose(res.signals["recon_tx"], res.signals["recon_rx"], atol=1e-12)

    # SNR estimate from book should be correct
    snr_est = float(pcm["snr_db_est"])
    assert abs(snr_est - (6.02 * n_bits + 1.76)) < 1e-12

@pytest.mark.parametrize("n_bits", [2, 4, 6])
def test_pcm_decode_meta_has_no_remainder_and_counts_match(n_bits):
    params = make_params(20)

    res = run(
        "sine",
        "PCM",
        params,
        Am=1.0, fm=5.0, duration=1.0,
        fs_mult=8, pcm_nbits=n_bits,
        linecode_scheme="NRZ-L",
    )
    assert_linecode_match(res)

    num_samples = int(res.meta["sampler"]["num_samples"])
    pcm_rx = res.meta["pcm_rx"]
    dec_meta = pcm_rx["decode_meta"]

    assert int(dec_meta["remainder_bits"]) == 0
    assert int(dec_meta["n_codewords"]) == num_samples

    idx_hat = np.asarray(pcm_rx["idx_hat"], dtype=int)
    q_hat = np.asarray(pcm_rx["q_hat"], dtype=float)
    assert len(idx_hat) == len(q_hat) == num_samples


# ==============================
# 5) DM: comparator + staircase invariants
# ==============================

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("delta", [0.02, 0.1, 0.25])
def test_dm_comparator_and_staircase_invariants(kind, delta):
    params = make_params(21)  # include odd Ns to exercise Manchester adjustment if selected later
    Am = 1.0
    fm = 6.0
    fs_mult = 8
    duration = 1.0

    res = run(
        kind,
        "DM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, dm_delta=delta,
        linecode_scheme="NRZ-L",
    )
    assert_linecode_match(res)

    dm = res.meta.get("dm", {})
    assert isinstance(dm, dict)
    assert float(dm["delta"]) == pytest.approx(float(delta))
    assert float(dm["est0"]) == pytest.approx(0.0)

    steps = dm.get("steps", [])
    assert isinstance(steps, list)

    # Extract sampled input and stair-before/after from steps
    xk = (
        np.array([row["Input sample x[k]"] for row in steps], dtype=float)
        if steps and "Input sample x[k]" in steps[0]
        else np.array([row.get("PAM sample") for row in steps], dtype=float)
    )
    stair_before = np.array([row["Stair (before)"] for row in steps], dtype=float)
    bits = np.array([row["DM bit"] for row in steps], dtype=int)
    stair_after = np.array([row["Stair (after)"] for row in steps], dtype=float)

    assert set(bits.tolist()).issubset({0, 1})

    # Comparator rule: bit=1 iff x >= stair_before
    assert np.all((bits == 1) == (xk >= stair_before))

    # Stair update: after = before ± Δ depending on bit
    delta_arr = np.where(bits == 1, +float(delta), -float(delta))
    assert np.allclose(stair_after, stair_before + delta_arr, atol=1e-12)

    # Bitstream length equals num_samples (1 bit/sample)
    num_samples = int(res.meta["sampler"]["num_samples"])
    assert len(res.bits["bitstream"]) == num_samples
    assert len(res.bits["decoded_bitstream"]) == num_samples

    # The plotted TX staircase includes est0 at t=0 then stair_after per sample
    stair_tx_plot = np.asarray(res.meta["stair_tx"]["x"], dtype=float)
    assert stair_tx_plot.shape[0] == stair_after.shape[0] + 1
    assert stair_tx_plot[0] == pytest.approx(0.0)
    assert np.allclose(stair_tx_plot[1:], stair_after, atol=1e-12)

    # RX staircase should match TX staircase (ideal line-coded channel)
    stair_rx_plot = np.asarray(res.meta["stair_rx"]["x"], dtype=float)
    assert np.allclose(stair_rx_plot, stair_tx_plot[: len(stair_rx_plot)], atol=1e-12)

    # Reconstructions on dense axis should match in ideal channel
    assert np.allclose(res.signals["recon_tx"], res.signals["recon_rx"], atol=1e-12)


# ==========================
# 6) duration=0 edge behavior
# ==========================

@pytest.mark.parametrize("technique", TECHNIQUES)
def test_duration_zero_returns_empty_signals_and_bits(technique):
    params = make_params(20)
    res = run(
        "sine",
        technique,
        params,
        Am=1.0, fm=5.0, duration=0.0,
        fs_mult=8, pcm_nbits=4, dm_delta=0.1,
        linecode_scheme="NRZ-L",
    )

    assert res.t.size == 0
    assert res.signals["m(t)"].size == 0
    assert res.signals["linecode"].size == 0
    assert res.signals["recon_tx"].size == 0
    assert res.signals["recon_rx"].size == 0

    assert res.bits["bitstream"] == []
    assert res.bits["decoded_bitstream"] == []
    assert_linecode_match(res)  # empty==empty should match

@pytest.mark.parametrize("technique", TECHNIQUES)
def test_duration_shorter_than_Ts_produces_one_sample(technique):
    params = make_params(20)
    fm = 1.0
    fs_mult = 2          # fs_samp = 2 Hz -> Ts = 0.5 s
    duration = 0.1       # < Ts -> only one sample at t=0

    res = run(
        "sine",
        technique,
        params,
        Am=1.0, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=3, dm_delta=0.1,
        linecode_scheme="NRZ-L",
    )

    assert int(res.meta["sampler"]["num_samples"]) == 1
    assert len(res.meta["sampled"]["t_s"]) == 1

    if technique == "PCM":
        assert len(res.bits["bitstream"]) == 3
    else:
        assert len(res.bits["bitstream"]) == 1

    assert_linecode_match(res)


# =====================================================
# 7) Deterministic fuzz tests (seeded, moderate sizes)
# =====================================================

def _rand_choice(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


@pytest.mark.parametrize("seed", list(range(40)))
def test_seeded_fuzz_pcm(seed):
    rng = random.Random(1000 + seed)

    params = make_params(_rand_choice(rng, [20, 21, 60]))
    kind = _rand_choice(rng, KINDS)
    linecode = _rand_choice(rng, LINECODES)

    Am = rng.uniform(0.5, 4.0)
    fm = rng.uniform(1.0, 20.0)
    fs_mult = _rand_choice(rng, [2, 4, 8, 16])
    n_bits = _rand_choice(rng, [2, 3, 4, 5, 6])
    duration = rng.uniform(0.2, 1.5)

    res = run(
        kind,
        "PCM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=n_bits,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, f"PCM fuzz mismatch seed={seed}")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]

    # Basic PCM size invariants
    num_samples = int(res.meta["sampler"]["num_samples"])
    assert len(res.bits["bitstream"]) == num_samples * n_bits


@pytest.mark.parametrize("seed", list(range(40)))
def test_seeded_fuzz_dm(seed):
    rng = random.Random(5000 + seed)

    params = make_params(_rand_choice(rng, [20, 21, 60]))
    kind = _rand_choice(rng, KINDS)
    linecode = _rand_choice(rng, LINECODES)

    Am = rng.uniform(0.5, 4.0)
    fm = rng.uniform(1.0, 20.0)
    fs_mult = _rand_choice(rng, [2, 4, 8, 16])
    delta = rng.uniform(0.01, 0.5)
    duration = rng.uniform(0.2, 1.5)

    res = run(
        kind,
        "DM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, dm_delta=delta,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, f"DM fuzz mismatch seed={seed}")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]

    # DM size invariants
    num_samples = int(res.meta["sampler"]["num_samples"])
    assert len(res.bits["bitstream"]) == num_samples


# ==========================================
# 8) Long-run stress tests
# ==========================================

def _assert_finite_core_signals(res):
    # Core arrays that must never produce NaN/Inf in long runs
    for k in ("m(t)", "linecode", "recon_tx", "recon_rx"):
        arr = res.signals.get(k, None)
        if arr is None:
            continue
        assert np.isfinite(arr).all(), f"{k} contains NaN/Inf in long-run test."


# -----------------------
# PCM long-run stress tests
# -----------------------

@pytest.mark.parametrize("linecode", LINECODES)
@pytest.mark.parametrize("pattern", [
    "max_rate_high_bits",     # large sample count + large bitstream
    "max_rate_low_bits",      # large sample count + coarser quantization
])
def test_long_run_pcm_all_linecodes(linecode, pattern):
    params = make_params(20)

    # Keep display axis manageable but generate many PAM samples.
    # fs_display is fixed in simulate_a2d; duration should not be too large.
    Am = 4.0
    fm = 50.0
    fs_mult = 32
    duration = 5.0

    if pattern == "max_rate_high_bits":
        n_bits = 6
    elif pattern == "max_rate_low_bits":
        n_bits = 2
    else:
        raise ValueError("Unknown pattern")

    res = run(
        "sine",
        "PCM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=n_bits,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, f"PCM long-run mismatch (linecode={linecode}, pattern={pattern})")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]

    _assert_finite_core_signals(res)

    # Size/shape sanity
    num_samples = int(res.meta["sampler"]["num_samples"])
    assert num_samples > 1000  # ensure it's actually "long-run"
    assert len(res.bits["bitstream"]) == num_samples * n_bits

    bit_len = int(res.meta["linecode"]["bit_len"])
    Ns_eff = ns_used(res, params)
    assert len(res.signals["linecode"]) == bit_len * Ns_eff
    assert len(res.meta["t_bits"]) == len(res.signals["linecode"])

    # Steps list should exist and be aligned (but don't iterate heavily)
    pcm = res.meta.get("pcm", {})
    assert isinstance(pcm.get("steps", None), list)
    assert len(pcm["steps"]) == num_samples


@pytest.mark.parametrize("kind", KINDS)
def test_long_run_pcm_all_waveforms_one_linecode(kind):
    # Stress all waveforms, but avoid multiplying test count by all linecodes.
    params = make_params(20)

    Am = 4.0
    fm = 50.0
    fs_mult = 32
    duration = 5.0
    n_bits = 4
    linecode = "NRZ-L"

    res = run(
        kind,
        "PCM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=n_bits,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, f"PCM long-run mismatch (kind={kind})")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]
    _assert_finite_core_signals(res)

    num_samples = int(res.meta["sampler"]["num_samples"])
    assert num_samples > 1000
    assert len(res.bits["bitstream"]) == num_samples * n_bits


def test_long_run_pcm_large_samples_per_bit():
    # One heavier waveform-length case: large Ns makes the linecode waveform much longer.
    # Keep sample count moderate to avoid huge step dict lists.
    params = make_params(60)

    Am = 3.0
    fm = 50.0
    fs_mult = 32
    duration = 3.0
    n_bits = 6
    linecode = "Bipolar-AMI"

    res = run(
        "sine",
        "PCM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, pcm_nbits=n_bits,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, "PCM long-run mismatch (large Ns)")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]
    _assert_finite_core_signals(res)

    bit_len = int(res.meta["linecode"]["bit_len"])
    Ns_eff = ns_used(res, params)
    assert len(res.signals["linecode"]) == bit_len * Ns_eff


# -----------------------
# DM long-run stress tests
# -----------------------

@pytest.mark.parametrize("linecode", LINECODES)
@pytest.mark.parametrize("pattern", [
    "slope_overload_like",    # very small delta (staircase can lag strongly)
    "large_delta",            # larger delta (more aggressive tracking)
])
def test_long_run_dm_all_linecodes(linecode, pattern):
    params = make_params(20)

    Am = 4.0
    fm = 50.0
    fs_mult = 32
    duration = 5.0

    if pattern == "slope_overload_like":
        delta = 0.02
    elif pattern == "large_delta":
        delta = 0.5
    else:
        raise ValueError("Unknown pattern")

    res = run(
        "sine",
        "DM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, dm_delta=delta,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, f"DM long-run mismatch (linecode={linecode}, pattern={pattern})")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]
    _assert_finite_core_signals(res)

    num_samples = int(res.meta["sampler"]["num_samples"])
    assert num_samples > 1000
    assert len(res.bits["bitstream"]) == num_samples  # DM: 1 bit per sample

    bit_len = int(res.meta["linecode"]["bit_len"])
    Ns_eff = ns_used(res, params)
    assert len(res.signals["linecode"]) == bit_len * Ns_eff

    # Staircase plot arrays should include initial estimate point
    stair_tx = np.asarray(res.meta["stair_tx"]["x"], dtype=float)
    stair_rx = np.asarray(res.meta["stair_rx"]["x"], dtype=float)
    assert len(stair_tx) >= num_samples  # should be num_samples+1 in the book-style build
    assert len(stair_rx) >= num_samples


def test_long_run_dm_square_wave_hard_case():
    # Square wave has sharp transitions; DM staircase may struggle but should remain stable and match bits.
    params = make_params(21)  # odd Ns + Manchester stress path
    Am = 3.0
    fm = 40.0
    fs_mult = 32
    duration = 4.0
    delta = 0.05
    linecode = "Manchester"

    res = run(
        "square",
        "DM",
        params,
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult, dm_delta=delta,
        linecode_scheme=linecode,
    )

    assert_linecode_match(res, "DM long-run mismatch (square/Manchester/odd Ns)")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]
    _assert_finite_core_signals(res)

    # Manchester must have even effective Ns
    assert ns_used(res, params) % 2 == 0


# -----------------------
# Manchester odd-Ns long-run (both techniques)
# -----------------------

@pytest.mark.parametrize("technique", ["PCM", "DM"])
def test_long_run_manchester_odd_Ns_both_techniques(technique):
    params = make_params(21)  # forces even-adjustment internally
    Am = 2.0
    fm = 50.0
    fs_mult = 32
    duration = 4.0

    kwargs = dict(
        Am=Am, fm=fm, duration=duration,
        fs_mult=fs_mult,
        linecode_scheme="Manchester",
    )
    if technique == "PCM":
        kwargs["pcm_nbits"] = 6
    else:
        kwargs["dm_delta"] = 0.1

    res = run("sine", technique, params, **kwargs)

    assert_linecode_match(res, f"{technique} long-run mismatch (Manchester odd Ns)")
    assert res.bits["decoded_bitstream"] == res.bits["bitstream"]
    _assert_finite_core_signals(res)

    assert ns_used(res, params) == 22

