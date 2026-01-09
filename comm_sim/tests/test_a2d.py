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
# NOTE
# ----
# Long-run stress tests are intentionally NOT included yet (per assignment instructions).
# A placeholder comment is kept at the end for adding them later.
#
# How to run
# ----------
#   pytest -q comm_sim/tests/test_a2d.py


from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import pytest

from utils import SimParams
from a2d import simulate_a2d


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
# 8) Long-run stress tests (to be added later)
# ==========================================
# TODO: Implement long-run stress tests for A2D (PCM + DM) later.
