import os
import random
from typing import Dict, List

import numpy as np
import pytest

from utils import SimParams
from d2d import simulate_d2d


# =========================
# Helpers / shared utilities
# =========================

SCHEMES_UI = [
    "NRZ-L",
    "NRZI",
    "Manchester",
    "Differential Manchester",
    "Bipolar-AMI",
    "Pseudoternary",
    "B8ZS",
    "HDB3",
]

# Extra aliases supported by implementation (sanity tests)
SCHEME_ALIASES = {
    "Differential Manchester": ["DiffManchester"],
    "Bipolar-AMI": ["AMI"],
}

BASIC_BITSTRS = [
    "",                  # empty input edge case
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
    "000000000",         # B8ZS overlap boundary (9 zeros)
    "00000000",          # exact B8ZS trigger
    "0000000000000000",  # 16 zeros
    "0000",              # HDB3 trigger
    "00000000",          # two HDB3 substitutions
    "100001",            # classic HDB3 parity-sensitive case
    "10000100001",       # multiple HDB3 substitutions separated by ones
]

def bits_from_str(s: str) -> List[int]:
    s = s.strip()
    if s == "":
        return []
    return [1 if c == "1" else 0 for c in s]

def make_params(Ns: int, Tb: float = 1.0) -> SimParams:
    return SimParams(
        fs=float(Ns) / float(Tb),
        Tb=float(Tb),
        samples_per_bit=int(Ns),
        Ac=1.0,     # not used for D2D
        fc=10.0,    # not used for D2D
    )

def run(bits: List[int], scheme: str, params: SimParams, **kwargs):
    return simulate_d2d(bits, scheme, params, **kwargs)

def enc_meta(res) -> Dict:
    return res.meta.get("encode", {}) if isinstance(res.meta, dict) else {}

def dec_meta(res) -> Dict:
    return res.meta.get("decode", {}) if isinstance(res.meta, dict) else {}

def substitutions(res) -> List[Dict]:
    return enc_meta(res).get("substitutions", []) or []

def hits(res) -> List[Dict]:
    return dec_meta(res).get("descramble_hits", []) or []

def ns_used_for_wave(scheme: str, params: SimParams, res) -> int:
    # For Manchester/Diff Manchester, encoder adjusts Ns to even and stores it.
    return int(enc_meta(res).get("Ns_adjusted_even", params.samples_per_bit))

def sample_ternary_levels_mid(wave: np.ndarray, Ns: int) -> List[int]:
    """
    Reconstruct per-bit ternary levels {-1,0,+1} from a clean repeated wave by mid-sample.
    Mirrors the decoder’s sampling logic at a high level.
    """
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    nbits = len(wave) // Ns
    out: List[int] = []
    for i in range(nbits):
        seg = wave[i * Ns:(i + 1) * Ns]
        mid = float(seg[len(seg) // 2]) if len(seg) else 0.0
        if abs(mid) < 0.5:
            out.append(0)
        else:
            out.append(+1 if mid > 0 else -1)
    return out

def assert_match(res, ctx: str = ""):
    assert res.meta.get("match", False), ctx or f"Mismatch in {res.meta.get('scheme')}"

def assert_wave_levels_subset(wave: np.ndarray, allowed: set, tol: float = 1e-9):
    if wave.size == 0:
        return
    uniq = set(np.unique(np.round(wave.astype(float), 12)))
    # tolerate tiny floating noise by rounding
    for v in uniq:
        assert any(abs(v - a) <= tol for a in allowed), f"Unexpected level {v} not in {allowed}"

def chunk_means(wave: np.ndarray, Ns: int) -> List[float]:
    nbits = len(wave) // Ns
    return [float(np.mean(wave[i * Ns:(i + 1) * Ns])) for i in range(nbits)]


# ============================
# 0) Basic API / error behavior
# ============================

def test_unknown_scheme_raises():
    params = make_params(20)
    with pytest.raises(ValueError):
        run([0, 1, 0], "NOT-A-SCHEME", params)

def test_ns_zero_is_invalid_for_decode_path():
    # simulate_d2d will try to sample with Ns; Ns=0 should error via internal sampling.
    params = make_params(1)
    params.samples_per_bit = 0
    with pytest.raises(Exception):
        run([0, 1, 1, 0], "NRZ-L", params)


# ==================================
# 1) Roundtrip grid (all schemes)
#    + odd Ns (Manchester adjustment)
# ==================================

@pytest.mark.parametrize("Ns", [20, 21])  # include odd Ns to force Manchester/Diff Manchester Ns_adjusted_even path
@pytest.mark.parametrize("scheme", SCHEMES_UI)
@pytest.mark.parametrize("bitstr", BASIC_BITSTRS)
def test_roundtrip_all_schemes_all_basic_patterns(Ns, scheme, bitstr):
    params = make_params(Ns)
    bits = bits_from_str(bitstr)

    # Use default kwargs except where a scheme depends on initial conditions (we cover those separately)
    res = run(bits, scheme, params)

    assert_match(res, f"{scheme} mismatch for bitstr='{bitstr}' Ns={Ns}")

    # Always preserve length
    assert len(res.bits["decoded"]) == len(bits)

    # Time axis and signal length sanity
    assert len(res.t) == len(res.signals["tx"])

    # Wave length equals bits * Ns_used (except empty)
    Ns_used = ns_used_for_wave(scheme, params, res)
    assert len(res.signals["tx"]) == len(bits) * Ns_used


# ============================
# 2) Scheme aliases (sanity)
# ============================

@pytest.mark.parametrize("Ns", [20, 21])
def test_aliases_match_canonical(Ns):
    params = make_params(Ns)
    bits = bits_from_str("01001100011")

    for canonical, aliases in SCHEME_ALIASES.items():
        res0 = run(bits, canonical, params)
        assert_match(res0, f"Canonical failed: {canonical}")

        for alias in aliases:
            # alias should roundtrip too
            res1 = run(bits, alias, params)
            assert_match(res1, f"Alias failed: {alias}")

            # encoded waveforms should be identical (same implementation branch)
            assert np.allclose(res0.signals["tx"], res1.signals["tx"])


# ==========================================
# 3) Level-set invariants (signal correctness)
# ==========================================

@pytest.mark.parametrize("Ns", [20, 21])
@pytest.mark.parametrize("bitstr", ["", "0", "1", "01010101", "111100001111"])
def test_level_sets_by_scheme(Ns, bitstr):
    params = make_params(Ns)
    bits = bits_from_str(bitstr)

    for scheme in SCHEMES_UI:
        res = run(bits, scheme, params)
        assert_match(res, f"Roundtrip failed for level-set test: {scheme}")

        tx = res.signals["tx"]
        if scheme in ("NRZ-L", "NRZI", "Manchester", "Differential Manchester"):
            assert_wave_levels_subset(tx, allowed={-1.0, +1.0})
        else:
            # AMI-family and pseudoternary use ternary levels
            assert_wave_levels_subset(tx, allowed={-1.0, 0.0, +1.0})


# ==========================================
# 4) Structural invariants (per-bit behavior)
# ==========================================

@pytest.mark.parametrize("Ns", [20, 21])
def test_nrzl_bit_to_level_convention(Ns):
    params = make_params(Ns)
    bits = bits_from_str("010011")
    res = run(bits, "NRZ-L", params)
    assert_match(res)

    Ns_used = ns_used_for_wave("NRZ-L", params, res)
    levels = sample_ternary_levels_mid(res.signals["tx"], Ns_used)
    # Book convention in code: 0 => +1, 1 => -1
    expected = [+1 if b == 0 else -1 for b in bits]
    assert levels == expected

@pytest.mark.parametrize("start", [-1, +1])
@pytest.mark.parametrize("Ns", [20, 21])
def test_nrzi_transition_rule(Ns, start):
    params = make_params(Ns)
    bits = bits_from_str("1010011100101001")
    res = run(bits, "NRZI", params, nrzi_start_level=start)
    assert_match(res)

    Ns_used = ns_used_for_wave("NRZI", params, res)
    levels = sample_ternary_levels_mid(res.signals["tx"], Ns_used)  # +/-1
    # Transition at start iff bit==1. Changes between consecutive bit levels equals number of ones after bit 0,
    # plus possible transition into first level relative to start_level when first bit is 1.
    changes = 0
    prev = start
    for i, b in enumerate(bits):
        if i == 0:
            if b == 1:
                assert levels[0] == -start
            else:
                assert levels[0] == start
        else:
            if levels[i] != levels[i-1]:
                changes += 1
    # For i>=1, a change occurs exactly when bit i is 1
    assert changes == sum(bits[1:])

@pytest.mark.parametrize("Ns", [20, 21])
def test_manchester_half_bit_structure(Ns):
    params = make_params(Ns)
    bits = bits_from_str("0100110011")
    res = run(bits, "Manchester", params)
    assert_match(res)

    Ns_used = ns_used_for_wave("Manchester", params, res)
    assert Ns_used % 2 == 0  # ensured

    h = Ns_used // 2
    tx = res.signals["tx"]
    for i, b in enumerate(bits):
        seg = tx[i*Ns_used:(i+1)*Ns_used]
        first = float(np.mean(seg[:h]))
        second = float(np.mean(seg[h:]))
        # IEEE-style in code: 1 = low->high, 0 = high->low
        if b == 1:
            assert first < second
        else:
            assert first > second
        # must always have mid-bit transition
        assert np.sign(first) != np.sign(second)

@pytest.mark.parametrize("start", [+1.0, -1.0])
@pytest.mark.parametrize("Ns", [20, 21])
def test_diff_manchester_start_transition_rule(Ns, start):
    params = make_params(Ns)
    bits = bits_from_str("0100110011")
    res = run(bits, "Differential Manchester", params, diff_start_level=start)
    assert_match(res)

    Ns_used = ns_used_for_wave("Differential Manchester", params, res)
    assert Ns_used % 2 == 0
    h = Ns_used // 2
    tx = res.signals["tx"]

    prev_last = start
    for i, b in enumerate(bits):
        seg = tx[i*Ns_used:(i+1)*Ns_used]
        first = float(np.mean(seg[:h]))
        second = float(np.mean(seg[h:]))

        start_transition = (np.sign(first) != np.sign(prev_last))
        # Convention in code: 0 => start transition, 1 => no start transition
        assert (start_transition and b == 0) or ((not start_transition) and b == 1)

        # Always mid-bit transition
        assert np.sign(first) != np.sign(second)

        prev_last = second


# =====================================================
# 5) Initial-condition sweeps (all relevant schemes)
# =====================================================

@pytest.mark.parametrize("bitstr", ["0", "1", "01001100011", "1010011100101001"])
@pytest.mark.parametrize("start", [-1, +1])
def test_nrzi_start_level_roundtrip(bitstr, start):
    params = make_params(20)
    res = run(bits_from_str(bitstr), "NRZI", params, nrzi_start_level=start)
    assert_match(res, f"NRZI start={start} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["0", "1", "01001100011", "1010011100101001"])
@pytest.mark.parametrize("start", [+1.0, -1.0])
def test_diff_manchester_start_level_roundtrip(bitstr, start):
    params = make_params(21)  # odd Ns also
    res = run(bits_from_str(bitstr), "Differential Manchester", params, diff_start_level=start)
    assert_match(res, f"DiffManchester start={start} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["01001100011", "111111", "1010101", "1000000001"])
@pytest.mark.parametrize("last", [-1, +1])
def test_ami_and_b8zs_last_pulse_init_roundtrip(bitstr, last):
    params = make_params(20)

    res1 = run(bits_from_str(bitstr), "Bipolar-AMI", params, last_pulse_init=last)
    assert_match(res1, f"AMI last={last} mismatch for {bitstr}")

    res2 = run(bits_from_str(bitstr), "B8ZS", params, last_pulse_init=last)
    assert_match(res2, f"B8ZS last={last} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["01001100011", "000000", "00100100", "0001000"])
@pytest.mark.parametrize("last0", [-1, +1])
def test_pseudoternary_last_zero_pulse_init_roundtrip(bitstr, last0):
    params = make_params(20)
    res = run(bits_from_str(bitstr), "Pseudoternary", params, last_zero_pulse_init=last0)
    assert_match(res, f"Pseudoternary last0={last0} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["100001", "0000", "00000000", "10000100001"])
@pytest.mark.parametrize("last", [-1, +1])
@pytest.mark.parametrize("parity", [0, 1])  # 0 even, 1 odd
def test_hdb3_initial_conditions_roundtrip(bitstr, last, parity):
    params = make_params(20)
    res = run(
        bits_from_str(bitstr),
        "HDB3",
        params,
        last_pulse_init=last,
        hdb3_nonzero_since_violation_init=parity,
    )
    assert_match(res, f"HDB3 last={last} parity={parity} mismatch for {bitstr}")


# ==================================================
# 6) B8ZS: exact pattern + counts + boundary behavior
# ==================================================

def _expected_b8zs_pattern(last_pulse: int) -> List[int]:
    v = -last_pulse
    b = -v
    return [0, 0, 0, v, b, 0, b, v]

@pytest.mark.parametrize("last", [-1, +1])
def test_b8zs_exact_8_zeros_inserts_expected_pattern(last):
    params = make_params(20)
    bits = bits_from_str("1" + "0"*8 + "1")
    res = run(bits, "B8ZS", params, last_pulse_init=last)
    assert_match(res)

    subs = substitutions(res)
    assert len(subs) == 1
    assert subs[0]["pos"] == 1

    Ns_used = ns_used_for_wave("B8ZS", params, res)
    tern = sample_ternary_levels_mid(res.signals["tx"], Ns_used)

    # The 8-zero window (bit positions 1..8) should become the exact B8ZS pattern
    assert tern[1:9] == _expected_b8zs_pattern(last)

@pytest.mark.parametrize("m", [0, 1, 7, 8, 9, 15, 16, 17, 24])
def test_b8zs_all_zeros_counts(m):
    params = make_params(20)
    bits = [0] * m
    res = run(bits, "B8ZS", params, last_pulse_init=-1)
    assert_match(res)

    expected = m // 8
    assert len(substitutions(res)) == expected
    assert len(hits(res)) == expected

@pytest.mark.parametrize("bitstr", ["0000000", "100000001", "1110000011110000"])
def test_b8zs_no_false_positive_without_8_consecutive_zeros(bitstr):
    params = make_params(20)
    res = run(bits_from_str(bitstr), "B8ZS", params, last_pulse_init=-1)
    assert_match(res)
    assert len(substitutions(res)) == 0
    assert len(hits(res)) == 0

@pytest.mark.parametrize("last", [-1, +1])
def test_b8zs_9_zeros_overlap_boundary(last):
    params = make_params(20)
    res = run([0]*9, "B8ZS", params, last_pulse_init=last)
    assert_match(res)
    assert len(substitutions(res)) == 1
    assert len(hits(res)) == 1


# ===============================================
# 7) HDB3: exact pattern + rule selection + counts
# ===============================================

def test_hdb3_rule_for_0000_depends_on_parity():
    params = make_params(20)
    bits = [0, 0, 0, 0]

    # even -> B00V (with last_pulse_init=-1 => B=+1, V=+1)
    res_even = run(bits, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res_even)
    subs_even = substitutions(res_even)
    assert len(subs_even) == 1
    assert subs_even[0]["rule"] == "B00V"
    assert subs_even[0]["pattern"] == [+1, 0, 0, +1]

    # odd -> 000V (V = last_pulse = -1)
    res_odd = run(bits, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=1)
    assert_match(res_odd)
    subs_odd = substitutions(res_odd)
    assert len(subs_odd) == 1
    assert subs_odd[0]["rule"] == "000V"
    assert subs_odd[0]["pattern"] == [0, 0, 0, -1]

def test_hdb3_parity_flips_rule_for_100001():
    params = make_params(20)
    bitstr = bits_from_str("100001")

    res_even = run(bitstr, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    res_odd  = run(bitstr, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=1)

    assert_match(res_even)
    assert_match(res_odd)

    rules_even = [s.get("rule") for s in substitutions(res_even)]
    rules_odd  = [s.get("rule") for s in substitutions(res_odd)]
    assert rules_even != rules_odd

@pytest.mark.parametrize("m", [0, 1, 3, 4, 5, 7, 8, 9, 16, 17])
@pytest.mark.parametrize("parity", [0, 1])
def test_hdb3_all_zeros_counts(m, parity):
    params = make_params(20)
    bits = [0] * m
    res = run(bits, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=parity)
    assert_match(res)

    expected = m // 4
    assert len(substitutions(res)) == expected
    assert len(hits(res)) == expected

def test_hdb3_no_false_positive_on_tricky_1001_like_patterns():
    # patterns that can look like B00V if decoder is sloppy
    params = make_params(20)
    for bitstr in ["1001", "110011001001", "101001001011"]:
        res = run(bits_from_str(bitstr), "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
        assert_match(res, f"HDB3 false-positive suspected on {bitstr}")


# ==============================================
# 8) Meta alignment sanity (scrambler <-> hits)
# ==============================================

@pytest.mark.parametrize("bitstr", ["1000000001", "1" + "0"*16 + "1", "0"*9, "0"*8])
def test_b8zs_meta_alignment_counts(bitstr):
    params = make_params(20)
    res = run(bits_from_str(bitstr), "B8ZS", params, last_pulse_init=-1)
    assert_match(res)
    assert len(substitutions(res)) == len(hits(res))

    # positions should align (both are bit/ternary indices)
    sub_pos = [s["pos"] for s in substitutions(res)]
    hit_pos = [h["pos"] for h in hits(res)]
    assert sub_pos == hit_pos

@pytest.mark.parametrize("bitstr", ["0000", "00000000", "100001", "10000100001", "111" + "0"*10 + "111"])
def test_hdb3_meta_alignment_counts(bitstr):
    params = make_params(20)
    res = run(bits_from_str(bitstr), "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res)
    assert len(substitutions(res)) == len(hits(res))

    sub_pos = [s["pos"] for s in substitutions(res)]
    hit_pos = [h["pos"] for h in hits(res)]
    assert sub_pos == hit_pos


# ==========================================
# 9) Fuzz tests (all schemes, deterministic)
# ==========================================

@pytest.mark.parametrize("Ns", [20, 21])
@pytest.mark.parametrize("scheme", SCHEMES_UI)
@pytest.mark.parametrize("seed", list(range(50)))
def test_seeded_fuzz_all_schemes(Ns, scheme, seed):
    params = make_params(Ns)
    rng = random.Random(seed)

    # Vary lengths to hit lots of boundary behavior
    n = rng.choice([0, 1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 127, 256, 512, 1024])
    bits = [rng.randint(0, 1) for _ in range(n)]

    res = run(bits, scheme, params)
    assert_match(res, f"{scheme} mismatch seed={seed} Ns={Ns} n={n}")

@pytest.mark.parametrize("seed", list(range(100)))
def test_seeded_fuzz_b8zs_hdb3_extra(seed):
    # Extra coverage on scramblers where corner cases tend to hide
    params = make_params(20)
    rng = random.Random(10_000 + seed)

    n = rng.choice([64, 128, 256, 512, 1024, 2048])
    bits = [rng.randint(0, 1) for _ in range(n)]

    res1 = run(bits, "B8ZS", params, last_pulse_init=-1)
    assert_match(res1, f"B8ZS mismatch seed={seed}")

    res2 = run(bits, "HDB3", params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res2, f"HDB3 mismatch seed={seed}")


# ==================================================
# 10) Long-run stress
# ==================================================

@pytest.mark.parametrize("scheme", ["B8ZS", "HDB3"])
def test_long_run_optional(scheme):
    params = make_params(20)
    # worst-case for scramblers: long zeros with occasional ones
    bits = [0] * 20_000 + [1] + [0] * 20_000 + [1] + [0] * 20_000

    if scheme == "B8ZS":
        res = run(bits, scheme, params, last_pulse_init=-1)
    else:
        res = run(bits, scheme, params, last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)

    assert_match(res, f"{scheme} long-run mismatch")
