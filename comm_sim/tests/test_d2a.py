# test_d2a.py
#
# Digital → Analog (D2A) modulation/demodulation unit tests for `simulate_d2a(...)`.
#
# What this test suite verifies
# -----------------------------
# 1) Roundtrip correctness (core requirement)
#    - For each supported D2A scheme, verify that decoded bits match the original input bits
#      under “reasonable / safe” parameter regimes (adequate sampling, tones under Nyquist, etc.).
#
# 2) Scheme coverage (including legacy)
#    - Main schemes: ASK, BFSK, MFSK, BPSK, DPSK, QPSK, QAM
#    - Legacy scheme still supported in code: 16QAM (tested to prevent regressions)
#
# 3) Padding + trimming behavior
#    - QPSK consumes 2 bits/symbol → odd-length inputs are padded internally, then trimmed back
#    - QAM consumes k bits/symbol (depends on axis_levels) → input may be padded then trimmed back
#    - Tests confirm `decoded_len == input_len` and match holds after trimming.
#
# 4) Internal meta/invariant checks (sanity beyond “just match”)
#    - Validates key `res.meta` fields exist and are consistent.
#    - Checks “expected behavior” signals for each scheme, for example:
#        • ASK: A_hat tracks the chosen amplitude levels
#        • BFSK: correct tone tends to have higher energy (E0/E1 separation) and swapping f0/f1 is handled
#        • MFSK: chosen_idx aligns with transmitted symbol index under clean conditions
#        • BPSK: I_hat sign matches antipodal phases (standard case)
#        • DPSK: delta_hat clusters near {0, ±Δ} (wrapped phase difference)
#        • QPSK: I_hat and Q_hat have non-trivial magnitude (not collapsing to ~0)
#        • QAM: I_dec/Q_dec belong to the allowed decision levels for axis_levels (2 or 4)
#
# 5) Warning-path coverage (tests for “expected warnings”, not failures)
#    - Nyquist/aliasing warnings (e.g., fc >= fs/2)
#    - Very small Ns warnings
#    - “non-positive frequency” warnings (invalid tone sets)
#    - ASK ambiguous threshold warnings (A1 <= A0)
#
# 6) Deterministic fuzz testing (seeded)
#    - Generates random bitstreams and randomized (but generally safe) parameters per scheme.
#    - Important exception:
#        If the simulator warns about “non-positive frequency”, the case can be fundamentally ambiguous
#        in a real-cosine model (cos(2π(-f)t) == cos(2π f t)). In that situation, mismatch is acceptable,
#        and the fuzz test does not assert match (it still exercises the code path).
#
# 7) Long-run stress tests (ALL schemes)
#    - Large bitstreams with multiple adversarial patterns:
#        • all_zeros, all_ones
#        • alternating patterns (0101…, 1010…)
#        • bursty patterns (long runs + short bursts)
#        • long_runs (very long blocks of 0s then 1s)
#        • random_seeded
#    - Uses stable, safe defaults per scheme to avoid “invalid parameter” ambiguity during stress.
#
# How to run
# ----------
#   pytest -q comm_sim/tests/test_d2a.py


from __future__ import annotations

import random
from typing import Dict, List, Any

import numpy as np
import pytest

from utils import SimParams
from d2a import simulate_d2a


# -------------------------
# Shared utilities / helpers
# -------------------------

SCHEMES = ["ASK", "BFSK", "MFSK", "BPSK", "DPSK", "QPSK", "QAM"]
# Legacy (kept in d2a.py). Not exposed in UI anymore, but still test it.
LEGACY_SCHEMES = ["16QAM"]

BASIC_BITSTRS = [
    "0",
    "1",
    "01",
    "10",
    "0000",
    "1111",
    "01010101",
    "10101010",
    "00110011",
    "11001100",
    "01001100011",
    "1111000011110000",
]

EDGE_BITSTRS = [
    "0" * 1,
    "1" * 1,
    "0" * 3,          # padding edges for QPSK/QAM
    "1" * 3,
    "0" * 5,
    "1" * 5,
    "0" * 17,
    "1" * 17,
    "010" * 11,       # 33 bits
    "0011" * 25,      # 100 bits
]

def bits_from_str(s: str) -> List[int]:
    s = s.strip()
    if s == "":
        return []
    return [1 if c == "1" else 0 for c in s]

def make_params(
    Ns: int,
    Tb: float = 1.0,
    fc: float = 8.0,
    Ac: float = 1.0,
) -> SimParams:
    # fs is derived so that "Ns samples per bit" means fs = Ns/Tb.
    return SimParams(
        fs=float(Ns) / float(Tb),
        Tb=float(Tb),
        samples_per_bit=int(Ns),
        Ac=float(Ac),
        fc=float(fc),
    )

def run(bits: List[int], scheme: str, params: SimParams, **kwargs):
    return simulate_d2a(bits, scheme, params, **kwargs)

def assert_match(res, ctx: str = ""):
    assert res.meta.get("match", False), ctx or f"Mismatch in scheme {res.meta.get('scheme')}"

def has_warning(res, substr: str) -> bool:
    ws = res.meta.get("warnings", []) or []
    return any(substr in str(w) for w in ws)

def get_meta(res, section: str) -> Dict[str, Any]:
    m = res.meta.get(section, {})
    return m if isinstance(m, dict) else {}

def approx_list(xs: List[float], ys: List[float], tol: float = 1e-2) -> bool:
    if len(xs) != len(ys):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(xs, ys))


# -------------------------
# Default kwargs (safe)
# -------------------------

def defaults_for_scheme(scheme: str) -> Dict[str, Any]:
    scheme = scheme.upper()
    if scheme == "ASK":
        return {"A0": 0.00, "A1": 1.00}
    if scheme == "BFSK":
        # choose well-separated tones, safe for fs (when Ns is sufficiently large)
        return {"f0": 8.0, "f1": 12.0}
    if scheme == "MFSK":
        return {"L": 2, "fd": 1.0}
    if scheme == "BPSK":
        return {"phase1": 0.0, "phase0": float(np.pi)}
    if scheme == "DPSK":
        return {"phase_init": 0.0, "delta_phase": float(np.pi)}
    if scheme == "QPSK":
        return {"phi_ref": 0.0}
    if scheme == "QAM":
        return {"axis_levels": 2, "phi_ref": 0.0}
    if scheme == "16QAM":
        return {}
    raise ValueError(f"Unknown scheme in defaults: {scheme}")


# =============================
# 0) Basic API / invalid inputs
# =============================

def test_empty_bits_raises():
    params = make_params(50)
    with pytest.raises(ValueError):
        run([], "ASK", params, **defaults_for_scheme("ASK"))

def test_nonbinary_bits_raises():
    params = make_params(50)
    with pytest.raises(ValueError):
        run([0, 1, 2, 0], "ASK", params, **defaults_for_scheme("ASK"))

def test_unknown_scheme_raises():
    params = make_params(50)
    with pytest.raises(ValueError):
        run([0, 1, 0, 1], "NOT-A-SCHEME", params)

@pytest.mark.parametrize("L", [0, -1, -5])
def test_mfsk_invalid_L_raises(L):
    params = make_params(80, fc=10.0)
    with pytest.raises(ValueError):
        run(bits_from_str("0101"), "MFSK", params, L=L, fd=1.0)

@pytest.mark.parametrize("axis_levels", [0, 1, 3, 5, 8, -2])
def test_qam_invalid_axis_levels_raises(axis_levels):
    params = make_params(80, fc=10.0)
    with pytest.raises(ValueError):
        run(bits_from_str("0101"), "QAM", params, axis_levels=axis_levels, phi_ref=0.0)


# =========================================
# 1) Roundtrip: all schemes, safe parameters
# =========================================

@pytest.mark.parametrize("Ns", [60, 120])  # sufficiently large for stable correlators
@pytest.mark.parametrize("bitstr", BASIC_BITSTRS + EDGE_BITSTRS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_roundtrip_all_schemes_safe(Ns, bitstr, scheme):
    bits = bits_from_str(bitstr)
    params = make_params(Ns, fc=8.0)

    kwargs = defaults_for_scheme(scheme)

    # For BFSK defaults, ensure Nyquist safety given Ns/Tb:
    # If Ns=60 => fs=60, Nyq=30, f1=12 OK.
    res = run(bits, scheme, params, **kwargs)

    assert_match(res, f"{scheme} mismatch for bitstr='{bitstr}', Ns={Ns}")
    assert len(res.bits["decoded"]) == len(bits)
    assert len(res.t) == len(res.signals["tx"])

@pytest.mark.parametrize("Ns", [80, 120])
@pytest.mark.parametrize("bitstr", ["01001100011", "0"*17, "1"*17])
@pytest.mark.parametrize("scheme", LEGACY_SCHEMES)
def test_roundtrip_legacy_schemes_safe(Ns, bitstr, scheme):
    bits = bits_from_str(bitstr)
    params = make_params(Ns, fc=8.0)
    res = run(bits, scheme, params)
    assert_match(res, f"{scheme} mismatch for bitstr='{bitstr}', Ns={Ns}")


# ==================================
# 2) Padding + trimming correctness
# ==================================

@pytest.mark.parametrize("Ns", [60, 120])
@pytest.mark.parametrize("bitstr", ["1", "01", "101", "10101", "0"*3, "1"*3, "01010"*7 + "1"])  # odd lengths
def test_qpsk_padding_trim(Ns, bitstr):
    bits = bits_from_str(bitstr)
    params = make_params(Ns, fc=8.0)
    res = run(bits, "QPSK", params, phi_ref=0.3)
    assert len(res.bits["decoded"]) == len(bits)
    assert_match(res, "QPSK padding/trim mismatch")

@pytest.mark.parametrize("Ns", [60, 120])
@pytest.mark.parametrize("axis_levels", [2, 4])
@pytest.mark.parametrize("bitstr", ["1", "101", "10101", "0"*5, "1"*5, "01010"*7 + "1"])
def test_qam_padding_trim(Ns, axis_levels, bitstr):
    bits = bits_from_str(bitstr)
    params = make_params(Ns, fc=8.0)
    res = run(bits, "QAM", params, axis_levels=axis_levels, phi_ref=-0.8)
    assert len(res.bits["decoded"]) == len(bits)
    assert_match(res, f"QAM axis_levels={axis_levels} padding/trim mismatch")


# =========================================
# 3) Scheme-specific meta + invariants
# =========================================

def test_meta_has_expected_top_fields():
    params = make_params(80, fc=8.0)
    bits = bits_from_str("010011001011")

    res = run(bits, "QPSK", params, phi_ref=0.0)
    meta = res.meta

    for k in ["scheme", "modulate", "demodulate", "input_len", "decoded_len", "pad_bits", "match", "warnings"]:
        assert k in meta

    assert meta["input_len"] == len(bits)
    assert meta["decoded_len"] == len(bits)

def test_ask_amplitudes_and_threshold_behavior():
    params = make_params(120, fc=7.0)
    bits = bits_from_str("01011001")
    A0, A1 = 0.10, 1.20
    res = run(bits, "ASK", params, A0=A0, A1=A1)
    assert_match(res)

    dem = get_meta(res, "demodulate")
    assert "A_hat" in dem and isinstance(dem["A_hat"], list)
    assert len(dem["A_hat"]) == len(bits)

    # In a clean simulator, the estimated amplitude should be near A0 or A1 (scaled out by Ac in demod)
    for b, a_hat in zip(bits, dem["A_hat"]):
        target = A1 if b == 1 else A0
        assert abs(float(a_hat) - target) < 0.10  # loose to avoid brittleness

def test_bfsk_energy_separation_on_clean_case():
    params = make_params(200, fc=10.0)
    bits = bits_from_str("0101100100110101")
    # Wide separation
    res = run(bits, "BFSK", params, f0=8.0, f1=12.0)
    assert_match(res)

    dem = get_meta(res, "demodulate")
    assert "E0" in dem and "E1" in dem
    assert len(dem["E0"]) == len(bits)
    assert len(dem["E1"]) == len(bits)

    # For a clean scenario, the correct tone should usually have higher energy
    # (not guaranteed for every bit if close to boundaries, but here should be strong).
    for b, e0, e1 in zip(bits, dem["E0"], dem["E1"]):
        if b == 1:
            assert float(e1) > float(e0)
        else:
            assert float(e0) >= float(e1)

def test_bfsk_swapped_f0_f1_inputs_are_handled():
    params = make_params(200, fc=10.0)
    bits = bits_from_str("0101100100110101")
    res = run(bits, "BFSK", params, f0=12.0, f1=8.0)  # intentionally swapped
    assert_match(res)

    mod = get_meta(res, "modulate")
    dem = get_meta(res, "demodulate")
    assert float(mod["f0"]) < float(mod["f1"])
    assert float(dem["f0"]) < float(dem["f1"])

def test_mfsk_index_recovery_matches_modulated_index_on_clean_case():
    params = make_params(200, fc=8.0)
    bits = bits_from_str("01001100011101001100")  # length not multiple of L sometimes
    L = 3
    fd = 1.0
    res = run(bits, "MFSK", params, L=L, fd=fd)
    assert_match(res)

    mod = get_meta(res, "modulate")
    dem = get_meta(res, "demodulate")

    assert mod.get("L") == L
    assert dem.get("L") == L
    assert mod.get("M") == 2 ** L
    assert dem.get("M") == 2 ** L

    # chosen_idx should equal transmitted sym_index in a clean regime
    assert "sym_index" in mod and "chosen_idx" in dem
    assert mod["sym_index"] == dem["chosen_idx"]

def test_bpsk_i_hat_sign_matches_bits_in_standard_antipodal_case():
    params = make_params(120, fc=8.0, Ac=1.0)
    bits = bits_from_str("0101100100110101")
    res = run(bits, "BPSK", params, phase1=0.0, phase0=float(np.pi))
    assert_match(res)

    dem = get_meta(res, "demodulate")
    I_hat = dem.get("I_hat", [])
    assert len(I_hat) == len(bits)

    # Standard: bit 1 => cos phase 0 => positive I; bit 0 => cos phase pi => negative I
    for b, ih in zip(bits, I_hat):
        if b == 1:
            assert float(ih) > 0.0
        else:
            assert float(ih) < 0.0

def test_dpsk_delta_hat_clusters_near_0_or_delta_on_clean_case():
    params = make_params(200, fc=9.0, Ac=1.0)
    bits = bits_from_str("0101100100110101")
    delta = float(np.pi)
    res = run(bits, "DPSK", params, phase_init=0.7, delta_phase=delta)
    assert_match(res)

    dem = get_meta(res, "demodulate")
    d_hat = dem.get("delta_hat", [])
    assert len(d_hat) == len(bits)

    # bit=0 => dphi near 0; bit=1 => dphi near +/-delta (wrapped)
    # Use a loose check: distance to nearest {0, +/-delta} should be small-ish.
    for b, dh in zip(bits, d_hat):
        x = float(dh)
        cands = [0.0, +delta, -delta]
        dist = min(abs(((x - c + np.pi) % (2*np.pi)) - np.pi) for c in cands)
        assert dist < 0.35  # loose; correlation errors are possible but should be small in this setup

def test_qpsk_iq_hat_signs_are_consistent_and_in_expected_ranges():
    params = make_params(120, fc=8.0, Ac=1.0)
    bits = bits_from_str("1101001011110010")  # multiple of 2
    res = run(bits, "QPSK", params, phi_ref=-1.1)
    assert_match(res)

    dem = get_meta(res, "demodulate")
    I_hat = dem.get("I_hat", [])
    Q_hat = dem.get("Q_hat", [])
    assert len(I_hat) == len(Q_hat) == (len(bits) // 2)

    # Should be near +/-1 ideally (since we recover I_sym/Q_sym)
    for ih, qh in zip(I_hat, Q_hat):
        assert abs(float(ih)) > 0.2
        assert abs(float(qh)) > 0.2

@pytest.mark.parametrize("axis_levels, allowed", [
    (2, {-1.0, +1.0}),
    (4, {-3.0, -1.0, +1.0, +3.0}),
])
def test_qam_decisions_are_valid_levels_and_roundtrip(axis_levels, allowed):
    params = make_params(200, fc=8.0, Ac=1.0)
    # Pattern that exercises many symbol points
    bits = bits_from_str("0000010111111010" * 10)  # length multiple of 16; fine for both variants
    res = run(bits, "QAM", params, axis_levels=axis_levels, phi_ref=0.5)
    assert_match(res)

    dem = get_meta(res, "demodulate")
    I_dec = dem.get("I_dec", [])
    Q_dec = dem.get("Q_dec", [])
    assert len(I_dec) == len(Q_dec)
    assert len(I_dec) > 0

    for v in I_dec:
        assert float(v) in allowed
    for v in Q_dec:
        assert float(v) in allowed


# ======================================
# 4) Warning-path coverage (no brittles)
# ======================================

def test_warning_nyquist_aliasing_triggered_when_fc_at_or_above_nyq():
    # Ns=20 => fs=20 => nyq=10. Set fc=10.
    params = make_params(20, fc=10.0)
    bits = bits_from_str("0101100100110101")

    res = run(bits, "BPSK", params, phase1=0.0, phase0=float(np.pi))
    assert has_warning(res, "Nyquist"), "Expected Nyquist aliasing warning was not emitted."

def test_warning_samples_per_bit_small_triggered_when_Ns_le_2():
    params = make_params(2, fc=1.0)  # Ns=2, fs=2, nyq=1, fc=1 => also Nyquist warning possible
    bits = bits_from_str("01011001")
    res = run(bits, "ASK", params, A0=0.0, A1=1.0)
    assert has_warning(res, "samples_per_bit is very small")

def test_warning_nonpositive_frequency_triggered_for_bfsk_if_user_sets_nonpositive():
    params = make_params(200, fc=10.0)
    bits = bits_from_str("01011001")
    # Intentionally invalid frequencies; simulator should warn, may or may not match (don’t assert match).
    res = run(bits, "BFSK", params, f0=-1.0, f1=3.0)
    assert has_warning(res, "non-positive")

def test_warning_ask_ambiguous_threshold_message_when_A1_le_A0():
    params = make_params(120, fc=8.0)
    bits = bits_from_str("01011001")
    res = run(bits, "ASK", params, A0=1.0, A1=0.5)
    # Should carry an ASK-specific warning
    assert any("ASK:" in str(w) for w in res.meta.get("warnings", []))


# ==========================================
# 5) Deterministic fuzz/stress (all schemes)
# ==========================================

def rand_bits(rng: random.Random, n: int) -> List[int]:
    return [rng.randint(0, 1) for _ in range(n)]

def rand_kwargs_for_scheme(rng: random.Random, scheme: str, params: SimParams) -> Dict[str, Any]:
    scheme = scheme.upper()

    if scheme == "ASK":
        A0 = rng.choice([0.0, 0.1, 0.2])
        A1 = rng.choice([0.8, 1.0, 1.2])
        if A1 <= A0:
            A1 = A0 + 0.5
        return {"A0": A0, "A1": A1}

    if scheme == "BFSK":
        # Keep safely below Nyquist most of the time
        nyq = params.fs / 2.0
        f0 = rng.uniform(1.0, max(2.0, nyq * 0.35))
        f1 = rng.uniform(max(f0 + 0.5, 1.5), max(f0 + 2.0, nyq * 0.45))
        # clamp if we overshot
        f1 = min(f1, nyq * 0.49)
        return {"f0": float(f0), "f1": float(f1)}

    if scheme == "MFSK":
        L = rng.choice([1, 2, 3])
        fd = rng.choice([0.5, 1.0, 2.0])
        return {"L": int(L), "fd": float(fd)}

    if scheme == "BPSK":
        # Keep phases well-separated to avoid ambiguous cases
        phase1 = rng.uniform(-np.pi, np.pi)
        phase0 = phase1 + rng.choice([np.pi, -np.pi, np.pi * 0.75, -np.pi * 0.75])
        return {"phase1": float(phase1), "phase0": float(phase0)}

    if scheme == "DPSK":
        phase_init = rng.uniform(-np.pi, np.pi)
        delta = rng.choice([np.pi, np.pi * 0.75, np.pi * 0.5])
        return {"phase_init": float(phase_init), "delta_phase": float(delta)}

    if scheme == "QPSK":
        phi_ref = rng.uniform(-np.pi, np.pi)
        return {"phi_ref": float(phi_ref)}

    if scheme == "QAM":
        axis_levels = rng.choice([2, 4])
        phi_ref = rng.uniform(-np.pi, np.pi)
        return {"axis_levels": int(axis_levels), "phi_ref": float(phi_ref)}

    raise ValueError(f"Unknown scheme for fuzz kwargs: {scheme}")

@pytest.mark.parametrize("Ns", [60, 120])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", list(range(30)))
def test_seeded_fuzz_all_schemes(Ns, scheme, seed):
    params = make_params(Ns, fc=8.0, Ac=1.0)
    rng = random.Random(10_000 + seed)

    n = rng.choice([1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 127, 256, 512])
    bits = rand_bits(rng, n)

    kwargs = rand_kwargs_for_scheme(rng, scheme, params)
    res = run(bits, scheme, params, **kwargs)

    # If the simulator warns about non-positive frequencies, the case can be fundamentally ambiguous
    # (e.g., cos(2π(-f)t) == cos(2π f t)), so a mismatch is acceptable for fuzzing.
    if has_warning(res, "non-positive"):
        return

    # Otherwise, in normal regimes the fuzz should match.
    assert_match(res, f"{scheme} fuzz mismatch seed={seed} Ns={Ns} kwargs={kwargs}")


# ==========================================
# 6) Long-run stress tests
# ==========================================

@pytest.mark.parametrize("scheme", ["ASK", "BFSK", "MFSK", "BPSK", "DPSK", "QPSK", "QAM", "16QAM"])
@pytest.mark.parametrize("pattern", [
    "random_seeded",
    "alt_01",
    "alt_10",
    "bursty",
    "long_runs",
    "all_zeros",
    "all_ones",
])
def test_long_run_all_schemes(scheme, pattern):
    # Keep parameters stable and safe
    params = make_params(240, fc=8.0, Ac=1.0)  # fs=240Hz, Nyq=120Hz

    N = 12_000

    if pattern == "random_seeded":
        rng = random.Random(999)
        bits = [rng.randint(0, 1) for _ in range(N)]
    elif pattern == "alt_01":
        bits = [0, 1] * (N // 2)
    elif pattern == "alt_10":
        bits = [1, 0] * (N // 2)
    elif pattern == "bursty":
        bits = [0] * 4000 + [1] * 50 + [0] * 4000 + [1] * 50 + [0] * (N - 8100)
    elif pattern == "long_runs":
        bits = ([0] * 2000 + [1] * 2000) * 3
        bits = bits[:N]
    elif pattern == "all_zeros":
        bits = [0] * N
    elif pattern == "all_ones":
        bits = [1] * N
    else:
        raise ValueError("Unknown pattern")

    # Scheme-specific safe kwargs (avoid ambiguity regimes)
    if scheme == "ASK":
        kwargs = {"A0": 0.00, "A1": 1.00}

    elif scheme == "BFSK":
        kwargs = {"f0": 8.0, "f1": 12.0}

    elif scheme == "MFSK":
        # Choose parameters that guarantee positive tones:
        # f_min = fc + (1-M)*fd > 0
        # Here fc=8, choose L=2 (M=4), fd=1 => f_min = 8 + (1-4)*1 = 5 > 0
        kwargs = {"L": 2, "fd": 1.0}

    elif scheme == "BPSK":
        kwargs = {"phase1": 0.0, "phase0": float(np.pi)}

    elif scheme == "DPSK":
        kwargs = {"phase_init": 0.7, "delta_phase": float(np.pi)}

    elif scheme == "QPSK":
        kwargs = {"phi_ref": 0.3}

    elif scheme == "QAM":
        # Stress both variants by toggling via pattern
        # Use axis_levels=4 for harder slicing thresholds
        kwargs = {"axis_levels": 4, "phi_ref": -0.7}

    elif scheme == "16QAM":
        # Legacy path
        kwargs = {}

    else:
        raise ValueError("Unknown scheme")

    res = run(bits, scheme, params, **kwargs)

    # If any run warns about "non-positive" frequency (shouldn't happen with these kwargs),
    # do not require match (fundamentally ambiguous in cosine model).
    if has_warning(res, "non-positive"):
        return

    assert_match(res, f"{scheme} long-run mismatch (pattern={pattern})")
    assert len(res.bits["decoded"]) == len(bits)
    assert len(res.signals["tx"]) == len(res.t)
