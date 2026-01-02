import random
import pytest

from utils import SimParams
from d2d import simulate_d2d


# -----------------------
# Helpers / shared params
# -----------------------

def bits_from_str(s: str):
    return [int(c) for c in s.strip()]

def run(bits, scheme, **kwargs):
    return simulate_d2d(bits, scheme, PARAMS, **kwargs)

def assert_match(res, msg=""):
    assert res.meta["match"], msg or f"Mismatch: {res.meta.get('scheme')}"

def enc(res):
    return res.meta.get("encode", {}) if isinstance(res.meta, dict) else {}

def dec(res):
    return res.meta.get("decode", {}) if isinstance(res.meta, dict) else {}

def subs(res):
    return enc(res).get("substitutions", []) or []

def hits(res):
    return dec(res).get("descramble_hits", []) or []


# Fast params for tests
Ns = 20
Tb = 1.0
PARAMS = SimParams(
    fs=Ns / Tb,
    Tb=Tb,
    samples_per_bit=Ns,
    Ac=1.0,   # required by SimParams, not used for d2d logic
    fc=10.0,
)


# -----------------------
# 1) Basic roundtrip grid
# -----------------------

ALL_SCHEMES = [
    "NRZ-L",
    "NRZI",
    "Manchester",
    "Differential Manchester",
    "Bipolar-AMI",
    "Pseudoternary",
    "B8ZS",
    "HDB3",
]

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

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("bitstr", BASIC_BITSTRS)
def test_all_schemes_basic_roundtrip(scheme, bitstr):
    res = run(bits_from_str(bitstr), scheme)
    assert_match(res, f"{scheme} mismatch for {bitstr}")


# -----------------------------------------------------
# 2) Initial-condition coverage (all combinations)
# -----------------------------------------------------

@pytest.mark.parametrize("bitstr", ["0", "1", "01001100011", "1010011100101001"])
@pytest.mark.parametrize("start", [-1, +1])
def test_nrzi_start_level_roundtrip(bitstr, start):
    res = run(bits_from_str(bitstr), "NRZI", nrzi_start_level=start)
    assert_match(res, f"NRZI start={start} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["0", "1", "01001100011", "1010011100101001"])
@pytest.mark.parametrize("start", [+1.0, -1.0])
def test_diff_manchester_start_level_roundtrip(bitstr, start):
    res = run(bits_from_str(bitstr), "Differential Manchester", diff_start_level=start)
    assert_match(res, f"DiffManchester start={start} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["01001100011", "111111", "1010101", "1000000001"])
@pytest.mark.parametrize("last", [-1, +1])
def test_ami_and_b8zs_preceding_one_polarity_roundtrip(bitstr, last):
    # Bipolar-AMI
    res1 = run(bits_from_str(bitstr), "Bipolar-AMI", last_pulse_init=last)
    assert_match(res1, f"AMI last={last} mismatch for {bitstr}")

    # B8ZS
    res2 = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=last)
    assert_match(res2, f"B8ZS last={last} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["01001100011", "000000", "00100100", "0001000"])
@pytest.mark.parametrize("last0", [-1, +1])
def test_pseudoternary_preceding_zero_polarity_roundtrip(bitstr, last0):
    res = run(bits_from_str(bitstr), "Pseudoternary", last_zero_pulse_init=last0)
    assert_match(res, f"Pseudoternary last0={last0} mismatch for {bitstr}")

@pytest.mark.parametrize("bitstr", ["100001", "0000", "00000000", "10000100001"])
@pytest.mark.parametrize("last", [-1, +1])
@pytest.mark.parametrize("parity", [0, 1])  # 0 even, 1 odd
def test_hdb3_initial_conditions_roundtrip(bitstr, last, parity):
    res = run(
        bits_from_str(bitstr),
        "HDB3",
        last_pulse_init=last,
        hdb3_nonzero_since_violation_init=parity,
    )
    assert_match(res, f"HDB3 last={last} parity={parity} mismatch for {bitstr}")


# -----------------------------------------------
# 3) B8ZS: exact trigger counts and boundaries
# -----------------------------------------------

def test_b8zs_7_zeros_no_trigger():
    res = run(bits_from_str("100000001"), "B8ZS", last_pulse_init=-1)
    assert_match(res)
    assert len(subs(res)) == 0
    assert len(hits(res)) == 0

def test_b8zs_exact_8_zeros_trigger_once():
    bitstr = "1000000001"
    for last in (-1, +1):
        res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=last)
        assert_match(res, f"B8ZS mismatch last={last}")
        assert len(subs(res)) == 1
        assert len(hits(res)) == 1

def test_b8zs_16_zeros_trigger_twice():
    bitstr = "1" + "0"*16 + "1"
    res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=-1)
    assert_match(res)
    assert len(subs(res)) == 2
    assert len(hits(res)) == 2

def test_b8zs_9_zeros_overlap_boundary():
    # regression for earlier bug
    bitstr = "0"*9
    for last in (-1, +1):
        res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=last)
        assert_match(res, f"B8ZS 9-zeros mismatch last={last}")
        assert len(subs(res)) == 1
        assert len(hits(res)) == 1

def test_b8zs_8_zeros_at_start_and_end():
    for bitstr in ["0"*8, "1" + "0"*8, "0"*8 + "1"]:
        for last in (-1, +1):
            res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=last)
            assert_match(res, f"B8ZS boundary mismatch {bitstr} last={last}")
            assert len(subs(res)) >= 1
            assert len(hits(res)) >= 1


# ----------------------------------------------------
# 4) HDB3: rule selection and multi-sub substitutions
# ----------------------------------------------------

def test_hdb3_parity_flips_rule_for_100001():
    bitstr = "100001"
    res_even = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    res_odd  = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=1)
    assert_match(res_even)
    assert_match(res_odd)
    rules_even = [s.get("rule") for s in subs(res_even)]
    rules_odd  = [s.get("rule") for s in subs(res_odd)]
    assert rules_even != rules_odd, f"Expected different rule selection even vs odd, got {rules_even} vs {rules_odd}"

def test_hdb3_two_substitutions_8_zeros():
    bitstr = "0"*8
    res = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res)
    assert len(subs(res)) == 2
    assert len(hits(res)) == 2

def test_hdb3_two_runs_separated_by_ones():
    bitstr = "10000100001"  # two groups of 4 zeros
    res = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res)
    assert len(subs(res)) == 2
    assert len(hits(res)) == 2

def test_hdb3_long_run_10_zeros_has_2_subs_and_leftover():
    bitstr = "111" + "0"*10 + "111"
    res = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res)
    assert len(subs(res)) == 2
    assert len(hits(res)) == 2


# ----------------------------------------------------------------
# 5) Descrambler false-positive regressions (critical)
# ----------------------------------------------------------------

def test_hdb3_no_false_positive_on_1001_patterns():
    # this pattern can look like B00V if decoder is wrong
    for bitstr in ["1001", "110011001001", "101001001011"]:
        res = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
        assert_match(res, f"HDB3 false-positive suspected on {bitstr}")

def test_b8zs_no_false_positive_without_8_zeros():
    for bitstr in ["1010101010", "0000000", "00000000"[:-1], "100000001", "1110000011110000"]:
        res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=-1)
        assert_match(res)
        assert len(subs(res)) == 0, f"Unexpected B8ZS substitution on {bitstr}"
        assert len(hits(res)) == 0, f"Unexpected B8ZS hit on {bitstr}"


# ---------------------------------------------------------
# 6) Encode/decode meta alignment sanity (B8ZS + HDB3)
# ---------------------------------------------------------

def test_b8zs_meta_alignment_counts():
    for bitstr in ["1000000001", "1" + "0"*16 + "1", "0"*9]:
        res = run(bits_from_str(bitstr), "B8ZS", last_pulse_init=-1)
        assert_match(res)
        assert len(subs(res)) == len(hits(res)), f"B8ZS subs/hits mismatch for {bitstr}"

def test_hdb3_meta_alignment_counts():
    for bitstr in ["0000", "00000000", "100001", "10000100001", "111" + "0"*10 + "111"]:
        res = run(bits_from_str(bitstr), "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
        assert_match(res)
        assert len(subs(res)) == len(hits(res)), f"HDB3 subs/hits mismatch for {bitstr}"


# -----------------------------------------
# 7) Big seeded stress tests (deterministic)
# -----------------------------------------

@pytest.mark.parametrize("seed", range(200))  # increase for more coverage
def test_b8zs_seeded_stress(seed):
    rng = random.Random(seed)
    n = 512
    bits = [rng.randint(0, 1) for _ in range(n)]
    res = run(bits, "B8ZS", last_pulse_init=-1)
    assert_match(res, f"B8ZS mismatch seed={seed}")

@pytest.mark.parametrize("seed", range(200))
def test_hdb3_seeded_stress(seed):
    rng = random.Random(seed)
    n = 512
    bits = [rng.randint(0, 1) for _ in range(n)]
    res = run(bits, "HDB3", last_pulse_init=-1, hdb3_nonzero_since_violation_init=0)
    assert_match(res, f"HDB3 mismatch seed={seed}")
