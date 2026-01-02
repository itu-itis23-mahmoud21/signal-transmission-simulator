from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SimParams:
    fs: float                 # sample rate (Hz)
    Tb: float                 # bit duration (s)
    samples_per_bit: int      # Ns (must be int)
    Ac: float                 # carrier amplitude
    fc: float                 # carrier frequency (Hz)


@dataclass
class SimResult:
    t: np.ndarray
    signals: Dict[str, np.ndarray]     # named waveforms
    bits: Dict[str, List[int]]         # named bit lists
    meta: Dict[str, Any]               # intermediate details


def bits_from_string(bitstr: str) -> List[int]:
    s = bitstr.strip().replace(" ", "")
    if not s:
        raise ValueError("Bitstring is empty.")
    if any(c not in "01" for c in s):
        raise ValueError("Bitstring must contain only 0 and 1.")
    return [1 if c == "1" else 0 for c in s]


def bits_to_string(bits: List[int]) -> str:
    return "".join("1" if b else "0" for b in bits)


def gen_random_bits(n: int, seed: Optional[int] = None) -> List[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2, size=n)]


def ensure_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def make_time_axis(num_samples: int, fs: float) -> np.ndarray:
    return np.arange(num_samples, dtype=float) / float(fs)


def bits_to_step(bits: List[int], Ns: int) -> np.ndarray:
    # Step plot helper: repeats each bit value Ns times
    return np.repeat(np.array(bits, dtype=float), Ns)


def fft_mag(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    # One-sided magnitude spectrum
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X)
    return f, mag
