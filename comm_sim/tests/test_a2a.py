# test_a2a.py
#
# Analog → Analog (A2A) modulation/demodulation unit tests for `simulate_a2a(...)`.
#
# Coverage goals (mirrors style of test_d2d/test_d2a/test_a2d):
#   1) Input validation & error paths
#   2) Output shape/key invariants + finiteness
#   3) Book-aligned modulation identities (AM/FM/PM, Stallings Ch.16.1 conventions)
#   4) Recovery quality in safe (noiseless) regimes for sine/triangle
#   5) Edge/corner behaviors (duration=0, zero indices, overmodulation flag, parameter sensitivity)
#
# NOTE: Long-run stress tests are intentionally omitted for now. A TODO is left at the end.

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import pytest

from utils import SimParams
from a2d import gen_message
from a2a import (
    PAD_CYCLES,
    simulate_a2a,
    am_modulate,
    fm_modulate,
    pm_modulate,
)


KINDS = ["sine", "triangle"]
SCHEMES = ["AM", "FM", "PM"]


# =========================
# Helpers / shared utilities
# =========================

def make_params(fs: float, *, fc: float = 200.0, Ac: float = 1.0) -> SimParams:
    # Tb / samples_per_bit are irrelevant for A2A, but SimParams requires them.
    return SimParams(fs=float(fs), Tb=1.0, samples_per_bit=1, Ac=float(Ac), fc=float(fc))


def padN_for(fs: float, fc: float, N: int) -> int:
    if N <= 0:
        return 0
    return max(8, int(round(float(PAD_CYCLES) * float(fs) / max(1.0, float(fc)))))


def metrics(x: np.ndarray, y: np.ndarray, *, guard_frac: float = 0.0) -> Tuple[float, float]:
    """
    Returns: (nrmse, corr) computed after removing mean, optionally ignoring edges.

    guard_frac: fraction of samples dropped from BOTH ends (e.g., 0.02 drops 2% start and 2% end)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = int(min(x.size, y.size))
    if n == 0:
        return 0.0, 1.0

    g = int(round(float(guard_frac) * n))
    if g > 0 and 2 * g < n:
        x = x[g:-g]
        y = y[g:-g]
    else:
        x = x[:n]
        y = y[:n]

    x0 = x - float(np.mean(x)) if x.size else x
    y0 = y - float(np.mean(y)) if y.size else y

    denom = float(np.linalg.norm(x0)) + 1e-12
    nrmse = float(np.linalg.norm(x0 - y0) / denom)

    sx = float(np.std(x0))
    sy = float(np.std(y0))
    if sx < 1e-12 or sy < 1e-12:
        corr = 1.0 if float(np.linalg.norm(x0 - y0)) < 1e-9 else 0.0
    else:
        corr = float(np.corrcoef(x0, y0)[0, 1])
    return nrmse, corr


def assert_finite(arr: np.ndarray, name: str):
    arr = np.asarray(arr, dtype=float)
    assert np.all(np.isfinite(arr)), f"{name} contains NaN/inf"


# ==================================
# 0) Basic API / validation behavior
# ==================================

def test_unknown_scheme_raises():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "NOT-A-SCHEME", params, Am=1.0, fm=5.0, duration=1.0)


def test_square_waveform_is_disallowed():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("square", "AM", params, Am=1.0, fm=5.0, duration=1.0)


def test_unknown_kind_raises_from_message_generator():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    with pytest.raises(Exception):
        simulate_a2a("sawtooth", "AM", params, Am=1.0, fm=5.0, duration=1.0)


@pytest.mark.parametrize("fm", [0.0, -1.0, -10.0])
def test_message_frequency_must_be_positive(fm):
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "AM", params, Am=1.0, fm=fm, duration=1.0)


def test_duration_negative_raises():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=-0.1)


@pytest.mark.parametrize("fs", [0.0, -1.0])
def test_fs_must_be_positive(fs):
    params = make_params(fs, fc=200.0, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0)


@pytest.mark.parametrize("fc", [0.0, -1.0])
def test_fc_must_be_positive(fc):
    params = make_params(2000.0, fc=fc, Ac=1.0)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0)


@pytest.mark.parametrize("Ac", [0.0, -1.0])
def test_Ac_must_be_positive(Ac):
    params = make_params(2000.0, fc=200.0, Ac=Ac)
    with pytest.raises(ValueError):
        simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0)


def test_scheme_and_kind_are_case_insensitive():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    res = simulate_a2a("SiNe", "fM", params, Am=1.0, fm=5.0, duration=0.5, nf=2 * np.pi * 50.0)
    assert res.meta["scheme"] == "FM"
    assert res.meta["kind"] == "sine"




# =========================================
# 1) Output structure / length invariants
# =========================================

@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("kind", KINDS)
def test_output_shapes_required_keys_and_finiteness(scheme, kind):
    fs = 2000.0
    fc = 200.0
    duration = 0.5
    params = make_params(fs, fc=fc, Ac=1.0)

    res = simulate_a2a(
        kind,
        scheme,
        params,
        Am=1.0,
        fm=5.0,
        duration=duration,
        na=0.5,
        nf=2 * np.pi * 50.0,
        np_=2.0,
    )

    N = int(round(duration * fs))
    assert res.t.size == N

    # Common signals
    for key in ["m(t)", "carrier", "tx", "recovered"]:
        assert key in res.signals

    # Scheme-specific signals
    if scheme == "AM":
        assert "envelope_est" in res.signals
        assert "envelope_theory" in res.signals
        assert "am" in res.meta
    elif scheme == "FM":
        assert "inst_freq" in res.signals
        assert "fm" in res.meta
    else:
        assert "phase_dev" in res.signals
        assert "pm" in res.meta

    # All signals must match length N and be finite
    for name, arr in res.signals.items():
        assert np.asarray(arr).size == N, f"Signal '{name}' length mismatch"
        assert_finite(arr, name)

    # Uniform time axis, starting at t=0
    if N > 0:
        assert abs(float(res.t[0]) - 0.0) < 1e-12
    if N > 1:
        dt = np.diff(res.t)
        assert np.allclose(dt, 1.0 / fs)

    # meta basics
    assert res.meta["scheme"] == scheme
    assert res.meta["kind"] == kind
    assert "summary" in res.meta and isinstance(res.meta["summary"], dict)


def test_duration_zero_returns_empty_arrays():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    res = simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=0.0, na=0.5)
    assert res.t.size == 0
    for _, v in res.signals.items():
        assert np.asarray(v).size == 0
    assert "summary" in res.meta


def test_padding_is_applied_and_crop_keeps_exact_window_length():
    fs = 2000.0
    fc = 200.0
    duration = 1.0
    params = make_params(fs, fc=fc, Ac=1.0)

    N = int(round(duration * fs))
    padN = padN_for(fs, fc, N)
    assert padN >= 8  # by design in padN_for

    res = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=duration, nf=2*np.pi*50)

    # returned time axis always exactly N (cropped window)
    assert res.t.size == N
    for _, v in res.signals.items():
        assert np.asarray(v).size == N


@pytest.mark.parametrize("kind", KINDS)
def test_message_has_expected_peak_amplitude(kind):
    fs = 4000.0
    fc = 400.0
    fm = 5.0
    duration = 1.0
    Am = 1.7
    params = make_params(fs, fc=fc, Ac=1.0)

    res = simulate_a2a(kind, "AM", params, Am=Am, fm=fm, duration=duration, na=0.5)
    m = res.signals["m(t)"]
    peak = float(np.max(np.abs(m)))

    # should be close to Am (triangle uses piecewise linear; allow a bit of tolerance)
    assert peak == pytest.approx(abs(Am), rel=0.02, abs=0.02)


# ==================================================
# 2) Book-aligned modulation identity checks (TX)
# ==================================================

def _rebuild_full_record(kind: str, *, fs: float, fc: float, Am: float, fm: float, duration: float):
    """
    Rebuild (t_full, m_full, padN, N, N_full) exactly as simulate_a2a does,
    so we can validate modulator identity even for FM (cumulative phase).
    """
    N = int(round(duration * fs))
    padN = padN_for(fs, fc, N)
    N_full = N + 2 * padN

    if N_full > 0:
        t_full = (np.arange(N_full, dtype=float) - padN) / fs
        m_full = gen_message(t_full, kind, Am, fm)
    else:
        t_full = np.array([], dtype=float)
        m_full = np.array([], dtype=float)

    return t_full, m_full, padN, N, N_full


def test_am_tx_matches_formula_using_full_record():
    fs = 4000.0
    fc = 400.0
    fm = 5.0
    duration = 1.0
    Am = 1.25
    na = 0.6
    Ac = 2.0

    params = make_params(fs, fc=fc, Ac=Ac)
    res = simulate_a2a("sine", "AM", params, Am=Am, fm=fm, duration=duration, na=na)

    t_full, m_full, padN, N, N_full = _rebuild_full_record("sine", fs=fs, fc=fc, Am=Am, fm=fm, duration=duration)
    m_peak = float(np.max(np.abs(m_full))) if m_full.size else 0.0
    x_full = (m_full / m_peak) if m_peak > 0 else np.zeros_like(m_full)

    s_full = am_modulate(x_full, t_full, Ac=Ac, fc=fc, na=na)
    expected_tx = s_full[padN:padN + N] if N > 0 else np.array([], dtype=float)

    assert np.allclose(res.signals["tx"], expected_tx, atol=1e-9, rtol=1e-7)

    # Theory envelope in-window should match Ac*(1+na*x)
    m = res.signals["m(t)"]
    m_peak_win = float(np.max(np.abs(m_full))) if m_full.size else 0.0  # same peak used in sim
    x = (m / m_peak_win) if m_peak_win > 0 else np.zeros_like(m)
    expected_env = Ac * (1.0 + na * x)
    assert np.allclose(res.signals["envelope_theory"], expected_env, atol=1e-9, rtol=1e-7)


def test_pm_tx_matches_formula_using_full_record():
    fs = 3000.0
    fc = 300.0
    fm = 3.0
    duration = 2.0
    Am = 1.0
    np_ = 1.5
    Ac = 1.0

    params = make_params(fs, fc=fc, Ac=Ac)
    res = simulate_a2a("triangle", "PM", params, Am=Am, fm=fm, duration=duration, np_=np_)

    t_full, m_full, padN, N, _ = _rebuild_full_record("triangle", fs=fs, fc=fc, Am=Am, fm=fm, duration=duration)
    s_full = pm_modulate(m_full, t_full, Ac=Ac, fc=fc, np_=np_)
    expected_tx = s_full[padN:padN + N] if N > 0 else np.array([], dtype=float)

    assert np.allclose(res.signals["tx"], expected_tx, atol=1e-9, rtol=1e-7)


def test_fm_tx_matches_integral_definition_using_full_record():
    fs = 5000.0
    fc = 500.0
    fm = 5.0
    duration = 1.0
    Am = 1.0
    nf = 2 * np.pi * 20.0
    Ac = 1.0

    params = make_params(fs, fc=fc, Ac=Ac)
    res = simulate_a2a("sine", "FM", params, Am=Am, fm=fm, duration=duration, nf=nf)

    t_full, m_full, padN, N, _ = _rebuild_full_record("sine", fs=fs, fc=fc, Am=Am, fm=fm, duration=duration)
    s_full = fm_modulate(m_full, t_full, Ac=Ac, fc=fc, nf=nf, fs=fs)
    expected_tx = s_full[padN:padN + N] if N > 0 else np.array([], dtype=float)

    assert np.allclose(res.signals["tx"], expected_tx, atol=1e-9, rtol=1e-7)


# ==========================================
# 3) Meta-field formula sanity checks
# ==========================================

def test_am_meta_fields_and_overmodulation_flag():
    params = make_params(2000.0, fc=200.0, Ac=2.0)

    res0 = simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0, na=0.8)
    am0 = res0.meta["am"]
    assert am0["overmodulated"] is False
    assert float(am0["modulation_index_mu"]) == pytest.approx(0.8)
    assert float(am0["bandwidth_hint_hz"]) == pytest.approx(2.0 * 5.0)

    res1 = simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0, na=1.2)
    am1 = res1.meta["am"]
    assert am1["overmodulated"] is True
    assert float(am1["modulation_index_mu"]) == pytest.approx(1.2)


def test_fm_meta_fields_match_formulas_for_sine():
    fs = 4000.0
    fc = 400.0
    fm = 5.0
    Am = 1.0
    nf = 2 * np.pi * 10.0  # => ΔF = 10 Hz for A_m=1
    params = make_params(fs, fc=fc, Ac=1.0)

    res = simulate_a2a("sine", "FM", params, Am=Am, fm=fm, duration=1.0, nf=nf)
    fm_meta = res.meta["fm"]

    delta_f = abs(nf) * abs(Am) / (2 * np.pi)
    beta = delta_f / fm
    bw = 2.0 * (delta_f + fm)

    assert float(fm_meta["delta_f_max_hz"]) == pytest.approx(delta_f, rel=1e-6, abs=1e-9)
    assert float(fm_meta["beta_index"]) == pytest.approx(beta, rel=1e-6, abs=1e-9)
    assert float(fm_meta["bw_carson_hz"]) == pytest.approx(bw, rel=1e-6, abs=1e-9)


def test_fm_peak_frequency_deviation_matches_meta_for_sine():
    fs = 5000.0
    fc = 500.0
    params = make_params(fs, fc=fc, Ac=1.0)

    Am = 1.0
    fm = 5.0
    nf = 2*np.pi*60.0
    res = simulate_a2a("sine", "FM", params, Am=Am, fm=fm, duration=1.0, nf=nf)

    delta_f_meta = float(res.meta["fm"]["delta_f_max_hz"])

    dev = res.signals["inst_freq"] - fc
    # ignore edges
    n = dev.size
    g = int(round(0.02 * n))
    dev_mid = dev[g:-g] if 2*g < n else dev

    peak_dev = float(np.max(np.abs(dev_mid)))
    assert peak_dev == pytest.approx(delta_f_meta, rel=0.05, abs=0.5)


def test_pm_meta_fields_match_formulas_for_sine():
    fs = 4000.0
    fc = 400.0
    fm = 5.0
    Am = 2.0
    np_ = 1.25
    params = make_params(fs, fc=fc, Ac=1.0)

    res = simulate_a2a("sine", "PM", params, Am=Am, fm=fm, duration=1.0, np_=np_)
    pm_meta = res.meta["pm"]

    delta_phi = abs(np_) * abs(Am)
    delta_f = abs(np_) * abs(Am) * fm
    bw = 2.0 * (delta_f + fm)

    assert float(pm_meta["delta_phi_max_rad"]) == pytest.approx(delta_phi, rel=1e-6, abs=1e-9)
    assert float(pm_meta["delta_f_max_hz_sine_approx"]) == pytest.approx(delta_f, rel=1e-6, abs=1e-9)
    assert float(pm_meta["bw_carson_hz_sine_approx"]) == pytest.approx(bw, rel=1e-6, abs=1e-9)


# ==========================================
# 4) Recovery quality (safe regimes)
# ==========================================

@pytest.mark.parametrize("kind", KINDS)
def test_am_recovery_is_high_quality(kind):
    fs = 4000.0
    params = make_params(fs, fc=400.0, Ac=1.0)

    res = simulate_a2a(kind, "AM", params, Am=1.0, fm=5.0, duration=1.0, na=0.6)

    m = res.signals["m(t)"]
    m_hat = res.signals["recovered"]

    nrmse_all, corr_all = metrics(m, m_hat, guard_frac=0.0)
    assert corr_all > 0.999
    assert nrmse_all < 0.03


@pytest.mark.parametrize("kind", KINDS)
def test_am_recovered_is_consistent_with_envelope_est(kind):
    fs = 4000.0
    fc = 400.0
    duration = 1.0
    Am = 1.2
    na = 0.6
    Ac = 1.5
    params = make_params(fs, fc=fc, Ac=Ac)

    res = simulate_a2a(kind, "AM", params, Am=Am, fm=5.0, duration=duration, na=na)

    env = res.signals["envelope_est"]
    m_hat = res.signals["recovered"]
    m = res.signals["m(t)"]

    # m_peak used by sim is max(|m_full|); in-window peak is close enough for this identity check.
    m_peak = float(np.max(np.abs(m))) + 1e-12

    m_hat_from_env = (env / Ac - 1.0) / na * m_peak

    nrmse, corr = metrics(m_hat, m_hat_from_env, guard_frac=0.02)
    assert corr > 0.999
    assert nrmse < 0.02


@pytest.mark.parametrize("kind", KINDS)
def test_fm_recovery_is_high_quality_with_reasonable_nf(kind):
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)

    nf = 2 * np.pi * 100.0
    res = simulate_a2a(kind, "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf)

    m = res.signals["m(t)"]
    m_hat = res.signals["recovered"]

    # Evaluate what the user sees (full window) — should still be decent.
    nrmse_all, corr_all = metrics(m, m_hat, guard_frac=0.0)
    assert corr_all > 0.97
    assert nrmse_all < 0.25

    # Interior region should be very strong (edge behavior can still exist at the first/last samples).
    nrmse_mid, corr_mid = metrics(m, m_hat, guard_frac=0.02)
    assert corr_mid > 0.995
    assert nrmse_mid < 0.10


def test_fm_inst_freq_mean_is_close_to_fc_for_sine():
    fs = 5000.0
    fc = 500.0
    params = make_params(fs, fc=fc, Ac=1.0)

    nf = 2*np.pi*80.0
    res = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf)

    inst = res.signals["inst_freq"]
    # ignore edges
    n = inst.size
    g = int(round(0.02 * n))
    inst_mid = inst[g:-g] if 2*g < n else inst
    assert float(np.mean(inst_mid)) == pytest.approx(fc, rel=0.01, abs=1.0)


@pytest.mark.parametrize("kind", KINDS)
def test_pm_recovery_is_high_quality_with_reasonable_np(kind):
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)

    np_ = 2.0
    res = simulate_a2a(kind, "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=np_)

    m = res.signals["m(t)"]
    m_hat = res.signals["recovered"]

    nrmse_all, corr_all = metrics(m, m_hat, guard_frac=0.0)
    assert corr_all > 0.97
    assert nrmse_all < 0.25

    nrmse_mid, corr_mid = metrics(m, m_hat, guard_frac=0.02)
    assert corr_mid > 0.995
    assert nrmse_mid < 0.10


@pytest.mark.parametrize("kind", KINDS)
def test_pm_phase_dev_equals_np_times_message(kind):
    fs = 4000.0
    fc = 400.0
    params = make_params(fs, fc=fc, Ac=1.0)

    np_ = 1.8
    res = simulate_a2a(kind, "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=np_)

    phase_dev = res.signals["phase_dev"]
    m = res.signals["m(t)"]
    assert np.allclose(phase_dev, np_ * m, atol=1e-9, rtol=1e-7)


# ==========================================
# 5) Expected corner behaviors (zero indices)
# ==========================================

def test_am_na_zero_returns_zero_recovered_and_constant_theory_envelope():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    res = simulate_a2a("sine", "AM", params, Am=1.0, fm=5.0, duration=1.0, na=0.0)

    assert np.allclose(res.signals["recovered"], 0.0)
    assert np.allclose(res.signals["envelope_theory"], 1.0, atol=1e-9)

    # Envelope estimate should be close to constant Ac (allow small numerical wiggle)
    env = res.signals["envelope_est"]
    assert float(np.mean(env)) == pytest.approx(1.0, rel=0.02, abs=0.02)
    assert float(np.std(env)) < 0.05


def test_fm_nf_zero_returns_zero_recovered():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    res = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=0.0)
    assert np.allclose(res.signals["recovered"], 0.0)


def test_pm_np_zero_returns_zero_recovered():
    params = make_params(2000.0, fc=200.0, Ac=1.0)
    res = simulate_a2a("sine", "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=0.0)
    assert np.allclose(res.signals["recovered"], 0.0)


def test_fm_negative_nf_flips_inst_freq_deviation_but_not_recovered():
    fs = 5000.0
    fc = 500.0
    params = make_params(fs, fc=fc, Ac=1.0)
    nf_pos = 2 * np.pi * 100.0
    nf_neg = -nf_pos

    res_pos = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf_pos)
    res_neg = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf_neg)

    # Recovered message should be the same (nf sign cancels in modulation+demodulation)
    m_hat_pos = res_pos.signals["recovered"]
    m_hat_neg = res_neg.signals["recovered"]
    nrmse, corr = metrics(m_hat_neg, m_hat_pos, guard_frac=0.02)
    assert corr > 0.99
    assert nrmse < 0.10

    # But instantaneous frequency deviation should flip sign
    fdev_pos = res_pos.signals["inst_freq"] - fc
    fdev_neg = res_neg.signals["inst_freq"] - fc
    nrmse_f, corr_f = metrics(fdev_neg, -fdev_pos, guard_frac=0.02)
    assert corr_f > 0.99
    assert nrmse_f < 0.10


def test_fm_tiny_nf_does_not_produce_nan_inf():
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)
    res = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=1e-6)
    for name, arr in res.signals.items():
        assert_finite(arr, f"FM tiny nf: {name}")

def test_pm_tiny_np_does_not_produce_nan_inf():
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)
    res = simulate_a2a("sine", "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=1e-6)
    for name, arr in res.signals.items():
        assert_finite(arr, f"PM tiny np: {name}")


# ==========================================
# 6) Parameter sensitivity checks (expected)
# ==========================================

def test_fm_recovery_generally_improves_with_larger_nf():
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)

    nf_small = 2 * np.pi * 10.0
    nf_big = 2 * np.pi * 100.0

    res_small = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf_small)
    res_big = simulate_a2a("sine", "FM", params, Am=1.0, fm=5.0, duration=1.0, nf=nf_big)

    nrmse_s, corr_s = metrics(res_small.signals["m(t)"], res_small.signals["recovered"], guard_frac=0.02)
    nrmse_b, corr_b = metrics(res_big.signals["m(t)"], res_big.signals["recovered"], guard_frac=0.02)

    # Not strictly monotone for all environments, but should generally improve.
    assert corr_b >= corr_s - 1e-3
    assert nrmse_b <= nrmse_s + 0.02


def test_pm_recovery_generally_improves_with_larger_np():
    fs = 5000.0
    params = make_params(fs, fc=500.0, Ac=1.0)

    np_small = 0.25
    np_big = 2.0

    res_small = simulate_a2a("sine", "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=np_small)
    res_big = simulate_a2a("sine", "PM", params, Am=1.0, fm=5.0, duration=1.0, np_=np_big)

    nrmse_s, corr_s = metrics(res_small.signals["m(t)"], res_small.signals["recovered"], guard_frac=0.02)
    nrmse_b, corr_b = metrics(res_big.signals["m(t)"], res_big.signals["recovered"], guard_frac=0.02)

    assert corr_b >= corr_s - 1e-3
    assert nrmse_b <= nrmse_s + 0.02


# ==========================================
# 7) Deterministic fuzz (safe random regimes)
# ==========================================

def test_deterministic_fuzz_safe_regime():
    rng = random.Random(1337)

    for _ in range(35):
        scheme = rng.choice(SCHEMES)
        kind = rng.choice(KINDS)

        fm = rng.choice([2.0, 3.0, 5.0, 8.0])
        cycles = rng.randint(2, 8)
        duration = cycles / fm  # exact integer number of cycles

        fs = rng.choice([2000.0, 3000.0, 5000.0])
        fc = rng.uniform(20.0 * fm, 0.20 * fs)

        Am = rng.uniform(0.2, 2.0)
        Ac = rng.uniform(0.5, 2.5)

        params = make_params(fs, fc=fc, Ac=Ac)

        kwargs: Dict[str, float] = {}
        if scheme == "AM":
            kwargs["na"] = rng.uniform(0.1, 0.9)
        elif scheme == "FM":
            kwargs["nf"] = 2 * np.pi * rng.uniform(50.0, 150.0)
        else:
            kwargs["np_"] = rng.uniform(0.8, 3.0)

        res = simulate_a2a(kind, scheme, params, Am=Am, fm=fm, duration=duration, **kwargs)

        for name, arr in res.signals.items():
            assert_finite(arr, f"{scheme}:{name}")

        # Quality checks (looser than deterministic cases)
        guard = 0.0 if scheme == "AM" else 0.02

        m_sig = res.signals["m(t)"]
        m_hat = res.signals["recovered"]

        if scheme == "AM":
            nrmse, corr = metrics(m_sig, m_hat, guard_frac=guard)
            assert corr > 0.995
            assert nrmse < 0.10
        else:
            n = int(min(m_sig.size, m_hat.size))
            g = int(round(guard * n))
            if g > 0 and 2 * g < n:
                ms = m_sig[g:-g]
                mh = m_hat[g:-g]
            else:
                ms = m_sig
                mh = m_hat

            ms0 = ms - float(np.mean(ms))
            mh0 = mh - float(np.mean(mh))
            gain = float(np.dot(ms0, mh0) / (np.dot(mh0, mh0) + 1e-12))

            m_hat_fit = gain * m_hat
            nrmse, corr = metrics(m_sig, m_hat_fit, guard_frac=guard)

            assert corr > 0.95
            thresh = 0.35 if kind == "triangle" else 0.31
            assert nrmse < thresh

            # Optional sanity: reject absurd scaling
            assert 0.25 < abs(gain) < 4.0


@pytest.mark.parametrize("scheme", ["FM", "PM"])
def test_near_nyquist_carrier_is_finite_even_if_recovery_degrades(scheme):
    fs = 2000.0
    fc = 0.45 * fs  # close to Nyquist
    params = make_params(fs, fc=fc, Ac=1.0)

    kwargs = {"nf": 2*np.pi*80.0} if scheme == "FM" else {"np_": 2.0}
    res = simulate_a2a("sine", scheme, params, Am=1.0, fm=5.0, duration=1.0, **kwargs)

    for name, arr in res.signals.items():
        assert_finite(arr, f"{scheme} near nyquist: {name}")


# ----------------------------
# Long-run stress tests (later)
# ----------------------------
# TODO: Add long-run stress tests (large fs*duration sweeps) after finalizing baseline correctness and CI budget.
