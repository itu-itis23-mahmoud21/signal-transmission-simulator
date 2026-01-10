from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from utils import SimParams, SimResult, make_time_axis, ensure_even

# ---------- Encoding helpers (Vectorized) ----------

def _nrzl_levels_vec(bits: np.ndarray) -> np.ndarray:
    # Book convention (Chapter 5): 0 = high (+1), 1 = low (-1)
    return np.where(bits == 1, -1, 1)


def _nrzi_levels_vec(bits: np.ndarray, start_level: int = -1) -> np.ndarray:
    # 1 causes transition, 0 causes no change.
    # Logic: New state = Old state * (-1 if bit=1 else 1)
    # We can model this as a cumulative product of multipliers.
    
    # Map bits to multipliers: 0 -> 1, 1 -> -1
    multipliers = np.where(bits == 1, -1, 1)
    
    # Cumulative product gives the state relative to the start
    # We multiply by start_level to offset the initial state correctly
    return start_level * np.cumprod(multipliers)


def _manchester_samples_vec(bits: np.ndarray, Ns: int) -> np.ndarray:
    Ns = ensure_even(Ns)
    h = Ns // 2
    n = len(bits)
    
    # IEEE-style: 1 = low->high (-1, +1), 0 = high->low (+1, -1)
    # We create a (n, 2) matrix representing the two halves of each bit
    
    # Default to 0 case: (+1, -1)
    patterns = np.ones((n, 2), dtype=float)
    patterns[:, 1] = -1.0 
    
    # Update for 1 case: (-1, +1)
    mask1 = (bits == 1)
    patterns[mask1, 0] = -1.0
    patterns[mask1, 1] = 1.0
    
    # Repeat each column h times to match sample rate
    # repeat(patterns, h, axis=1) would give h copies of col0 then h copies of col1
    # We want interleaved: col0 repeated h times, then col1 repeated h times per row
    # Easier to just repeat_elements on the flattened pattern if we construct it right.
    # Actually, np.repeat on axis 1 gives [c0, c0, c1, c1], we want that per bit.
    
    full_pattern = np.repeat(patterns, h, axis=1)
    return full_pattern.ravel()


def _diff_manchester_samples_vec(bits: np.ndarray, Ns: int, start_level: float = +1.0) -> np.ndarray:
    Ns = ensure_even(Ns)
    h = Ns // 2
    n = len(bits)
    
    # Logic from analysis:
    # There is ALWAYS a transition in the middle.
    # Bit 0: Transition at start.
    # Bit 1: No transition at start.
    
    # Let 'curr_start' be the level at the beginning of a bit period.
    # Let 'curr_first' be the level of the first half of the bit.
    # Let 'curr_second' be the level of the second half.
    
    # Since mid-transition is mandatory, curr_second = -curr_first.
    # The level entering the NEXT bit is curr_second.
    # So curr_start[i+1] = curr_second[i] = -curr_first[i].
    
    # Transition rules for curr_first based on curr_start:
    # if b=0 (trans): curr_first = -curr_start
    # if b=1 (no trans): curr_first = curr_start
    
    # Substitute curr_start:
    # curr_start[i+1] = -1 * ( -1 if b=0 else 1 ) * curr_start[i]
    #                 = (1 if b=0 else -1) * curr_start[i]
    
    # 1. Calculate the sequence of start levels
    # Multipliers for start_levels recurrence:
    # if b=0: mult = 1. if b=1: mult = -1.
    seq_mults = np.where(bits == 0, 1.0, -1.0)
    
    # Shifted cumprod for starts: start[0]=init, start[1]=init*m[0]...
    # Prepend 1.0 to multipliers to represent the initial state 
    factors = np.r_[1.0, seq_mults[:-1]] 
    curr_starts = start_level * np.cumprod(factors)
    
    # 2. Calculate first halves
    # if b=0: first = -start. if b=1: first = start.
    first_halves = np.where(bits == 0, -1.0, 1.0) * curr_starts
    
    # 3. Calculate second halves (always opposite of first)
    second_halves = -first_halves
    
    # 4. Construct wave
    combined = np.stack((first_halves, second_halves), axis=1) # (N, 2)
    return np.repeat(combined, h, axis=1).ravel()


def _ami_levels_vec(bits: np.ndarray, last_pulse_init: int = -1) -> np.ndarray:
    # 0 -> 0
    # 1 -> alternating polarity
    
    out = np.zeros_like(bits, dtype=int)
    ones_mask = (bits == 1)
    
    # Count how many ones we've seen to determine polarity
    # cumsum on boolean mask gives 1, 2, 3... at the positions of ones
    ones_count = np.cumsum(ones_mask)
    
    # Polarity sequence: 
    # if last_init = -1 (neg), first one should be +1.
    # (-1)^k matches alternation.
    # We want: last_pulse_init * (-1) * (-1 if k is odd else 1)?
    # Simpler: last_pulse_init * (-1)^count
    # e.g., init=-1. count=1 -> -1*-1 = +1. Correct.
    # e.g., init=+1. count=1 -> +1*-1 = -1. Correct.
    
    polarities = last_pulse_init * (np.power(-1, ones_count))
    
    # Apply polarities only where bits are 1
    out[ones_mask] = polarities[ones_mask]
    return out


def _pseudoternary_levels_vec(bits: np.ndarray, last_zero_pulse_init: int = -1) -> np.ndarray:
    # 1 -> 0
    # 0 -> alternating polarity
    
    out = np.zeros_like(bits, dtype=int)
    zeros_mask = (bits == 0)
    
    zeros_count = np.cumsum(zeros_mask)
    polarities = last_zero_pulse_init * (np.power(-1, zeros_count))
    
    out[zeros_mask] = polarities[zeros_mask]
    return out


# Note: B8ZS and HDB3 are inherently sequential state machines with look-ahead.
# Pure vectorization is extremely complex and hard to read/maintain.
# The "Optimized" approach here uses pre-allocated arrays and direct indexing
# to avoid the overhead of Python list.append() and dynamic resizing.

def _b8zs_scramble_ami_opt(bits: List[int], last_pulse_init: int = -1) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(bits)
    out = np.zeros(n, dtype=int)
    meta = {"substitutions": []}
    
    # Convert input to array for fast slicing
    bits_arr = np.array(bits, dtype=int)
    
    last_pulse = last_pulse_init
    i = 0
    
    while i < n:
        if bits_arr[i] == 0:
            # Lookahead check for 8 zeros
            if i + 8 <= n and np.all(bits_arr[i:i+8] == 0):
                v = last_pulse
                b = -v
                # Pattern: 0 0 0 V B 0 B V
                pattern = np.array([0, 0, 0, v, b, 0, b, v], dtype=int)
                
                out[i:i+8] = pattern
                meta["substitutions"].append({
                    "pos": i, "type": "B8ZS", 
                    "pattern": pattern.tolist(), # Convert back to list for JSON/UI compatibility
                    "last_pulse": int(last_pulse)
                })
                i += 8
                continue
            else:
                out[i] = 0
        else:
            last_pulse *= -1
            out[i] = last_pulse
            
        i += 1
        
    return out, meta


def _hdb3_scramble_ami_opt(
    bits: List[int],
    last_pulse_init: int = -1,
    nonzero_since_violation_init: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    n = len(bits)
    out = np.zeros(n, dtype=int)
    meta = {"substitutions": []}
    
    last_pulse = last_pulse_init
    nonzero_since_violation = int(nonzero_since_violation_init)
    
    bits_arr = np.array(bits, dtype=int)
    i = 0
    
    while i < n:
        if bits_arr[i] == 0:
            if i + 4 <= n and np.all(bits_arr[i:i+4] == 0):
                # We have 4 zeros, must substitute
                
                if nonzero_since_violation % 2 == 1:
                    # Odd number of 1s since last sub -> 000V
                    v = last_pulse # Violation = same polarity as previous pulse
                    pattern = np.array([0, 0, 0, v], dtype=int)
                    rule = "000V"
                    
                    out[i:i+4] = pattern
                    meta["substitutions"].append({
                        "pos": i, "type": "HDB3", 
                        "pattern": pattern.tolist(), "rule": rule,
                        "last_pulse_before": int(last_pulse),
                        "count_since_last_sub": nonzero_since_violation
                    })
                    
                    # Last pulse is now V
                    last_pulse = v
                
                else:
                    # Even number -> B00V
                    b = -last_pulse # Valid = opposite polarity
                    v = b           # Violation = same as valid (so same as B)
                    pattern = np.array([b, 0, 0, v], dtype=int)
                    rule = "B00V"
                    
                    out[i:i+4] = pattern
                    meta["substitutions"].append({
                        "pos": i, "type": "HDB3", 
                        "pattern": pattern.tolist(), "rule": rule,
                        "last_pulse_before": int(last_pulse),
                        "count_since_last_sub": nonzero_since_violation
                    })
                    
                    # Last pulse is now V
                    last_pulse = v
                
                nonzero_since_violation = 0
                i += 4
                continue
            else:
                out[i] = 0
        else:
            last_pulse *= -1
            out[i] = last_pulse
            nonzero_since_violation += 1
            
        i += 1
        
    return out, meta


def _ternary_to_wave_vec(levels: np.ndarray, Ns: int) -> np.ndarray:
    if levels.size == 0:
        return np.array([], dtype=float)
    return np.repeat(levels.astype(float), Ns)


# ---------- Decoding helpers (Vectorized) ----------

def _sample_bit_levels_vec(wave: np.ndarray, Ns: int) -> np.ndarray:
    if Ns <= 0:
        raise ValueError("Ns must be positive.")
    
    n_bits = len(wave) // Ns
    # Reshape to (n_bits, Ns) to process all bits in parallel
    reshaped = wave[:n_bits*Ns].reshape(n_bits, Ns)
    
    # Take the middle sample of each bit period
    mids = reshaped[:, Ns // 2]
    
    # Thresholding logic: abs < 0.5 -> 0, else sign
    # np.sign gives 0 for 0, but our threshold is 0.5
    out = np.zeros(n_bits, dtype=int)
    out[mids > 0.5] = 1
    out[mids < -0.5] = -1
    
    return out


def _decode_manchester_vec(wave: np.ndarray, Ns: int) -> List[int]:
    Ns = ensure_even(Ns)
    h = Ns // 2
    n_bits = len(wave) // Ns
    
    reshaped = wave[:n_bits*Ns].reshape(n_bits, Ns)
    
    # Compare mean of first half vs second half
    # Axis 1 is the time within the bit
    first_mean = np.mean(reshaped[:, :h], axis=1)
    second_mean = np.mean(reshaped[:, h:], axis=1)
    
    # 1: low->high (first < second), 0: high->low (first > second)
    bits = np.where(first_mean < second_mean, 1, 0)
    return bits.tolist()


def _decode_diff_manchester_vec(wave: np.ndarray, Ns: int, start_level: float = +1.0) -> List[int]:
    Ns = ensure_even(Ns)
    h = Ns // 2
    n_bits = len(wave) // Ns
    
    reshaped = wave[:n_bits*Ns].reshape(n_bits, Ns)
    
    # We rely on the level of the first half of the current bit
    # vs the level of the second half of the previous bit (which equals the start level).
    
    first_halves = np.mean(reshaped[:, :h], axis=1)
    
    # We also need the "previous last levels". 
    # For bit 0, this is start_level. 
    # For bit i > 0, this is the second half of bit i-1.
    second_halves = np.mean(reshaped[:, h:], axis=1)
    
    # Shift second_halves to get "prev_last" for each index
    # prev_last[0] = start_level
    # prev_last[1] = second_halves[0]
    prev_last = np.r_[start_level, second_halves[:-1]]
    
    # Transition logic:
    # If first_half has different sign than prev_last => Transition => Bit 0
    # If same sign => No Transition => Bit 1
    
    # Use sign() to be robust against amplitude variations
    has_transition = (np.sign(first_halves) != np.sign(prev_last))
    
    bits = np.where(has_transition, 0, 1)
    return bits.tolist()


def _descramble_b8zs_opt(tern: np.ndarray, last_pulse_init: int = -1) -> Tuple[List[int], Dict[str, Any]]:
    # Iterative approach is best for decoding scan with variable jumps
    out_bits = []
    meta = {"descramble_hits": []}
    last_pulse = last_pulse_init
    i = 0
    n = len(tern)
    
    while i < n:
        # Check for B8ZS signature: 0 0 0 V B 0 B V
        # Optimization: Check ternary value at i+3 (V) first
        if i + 8 <= n:
            v = tern[i+3]
            # V must be nonzero and same polarity as last valid pulse
            if v != 0 and v == last_pulse: 
                 # Check the full pattern
                 chunk = tern[i:i+8]
                 # Expected: 0,0,0,V,-V,0,-V,V
                 if (chunk[0]==0 and chunk[1]==0 and chunk[2]==0 and chunk[5]==0):
                     b = chunk[4]
                     if b == -v and chunk[6] == b and chunk[7] == v:
                         # Match found
                         out_bits.extend([0]*8)
                         meta["descramble_hits"].append({
                             "pos": i, "type": "B8ZS",
                             "chunk": chunk.tolist(),
                             "v": int(v), "b": int(b),
                             "last_pulse": int(last_pulse)
                         })
                         # last_pulse remains v
                         i += 8
                         continue

        val = tern[i]
        if val == 0:
            out_bits.append(0)
        else:
            out_bits.append(1)
            last_pulse = val
        i += 1
        
    return out_bits, meta


def _descramble_hdb3_opt(tern: np.ndarray, last_nonzero_init: int | None = None) -> Tuple[List[int], Dict[str, Any]]:
    out_bits = []
    meta = {"descramble_hits": []}
    last_nonzero = last_nonzero_init
    
    i = 0
    n = len(tern)

    while i < n:
        # Lookahead for 4-symbol window
        if i + 4 <= n:
            w = tern[i:i+4]
            
            # Case 1: 000V (V matches last_nonzero)
            if w[0] == 0 and w[1] == 0 and w[2] == 0:
                v = w[3]
                if v != 0 and last_nonzero is not None and v == last_nonzero:
                    out_bits.extend([0, 0, 0, 0])
                    meta["descramble_hits"].append({
                        "pos": i, "type": "HDB3", "rule": "000V",
                        "window": w.tolist(), "last_nonzero": int(last_nonzero)
                    })
                    last_nonzero = v
                    i += 4
                    continue
            
            # Case 2: B00V (B opposite last_nonzero, V matches B)
            if w[1] == 0 and w[2] == 0:
                b = w[0]
                v = w[3]
                if b != 0 and v != 0:
                    if last_nonzero is not None and b == -last_nonzero and v == b:
                        out_bits.extend([0, 0, 0, 0])
                        meta["descramble_hits"].append({
                            "pos": i, "type": "HDB3", "rule": "B00V",
                            "window": w.tolist(), "last_nonzero": int(last_nonzero)
                        })
                        last_nonzero = v
                        i += 4
                        continue

        # Normal bit decoding
        val = tern[i]
        if val == 0:
            out_bits.append(0)
        else:
            out_bits.append(1)
            last_nonzero = val
        i += 1
        
    return out_bits, meta


# ---------- Public API ----------

def line_encode(
    bits: List[int],
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,       # for AMI/B8ZS/HDB3 (preceding '1' polarity)
    last_zero_pulse_init: int = -1,  # for Pseudoternary (preceding '0' polarity)
    hdb3_nonzero_since_violation_init: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}
    
    # Convert input list to numpy array for vectorization
    bits_arr = np.array(bits, dtype=int)

    if scheme == "NRZ-L":
        levels = _nrzl_levels_vec(bits_arr)
        return _ternary_to_wave_vec(levels, Ns), meta

    if scheme == "NRZI":
        levels = _nrzi_levels_vec(bits_arr, start_level=nrzi_start_level)
        return _ternary_to_wave_vec(levels, Ns), meta

    if scheme == "Manchester":
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _manchester_samples_vec(bits_arr, Ns), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _diff_manchester_samples_vec(bits_arr, Ns, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _ami_levels_vec(bits_arr, last_pulse_init=last_pulse_init)
        return _ternary_to_wave_vec(levels, Ns), meta

    if scheme == "Pseudoternary":
        levels = _pseudoternary_levels_vec(bits_arr, last_zero_pulse_init=last_zero_pulse_init)
        return _ternary_to_wave_vec(levels, Ns), meta

    if scheme == "B8ZS":
        # Pass original list or array; function handles conversion internally
        levels, smeta = _b8zs_scramble_ami_opt(bits, last_pulse_init=last_pulse_init)
        meta.update(smeta)
        return _ternary_to_wave_vec(levels, Ns), meta

    if scheme == "HDB3":
        levels, smeta = _hdb3_scramble_ami_opt(
            bits,
            last_pulse_init=last_pulse_init,
            nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
        )
        meta.update(smeta)
        return _ternary_to_wave_vec(levels, Ns), meta

    raise ValueError(f"Unknown scheme: {scheme}")


def line_decode(
    wave: np.ndarray,
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,  # unused in decode; for API symmetry
) -> Tuple[List[int], Dict[str, Any]]:
    
    Ns = int(params.samples_per_bit)
    meta: Dict[str, Any] = {"scheme": scheme}

    if scheme == "NRZ-L":
        levels = _sample_bit_levels_vec(wave, Ns)
        # +1 -> 0, -1 -> 1
        bits = np.where(levels > 0, 0, 1)
        return bits.tolist(), meta

    if scheme == "NRZI":
        levels = _sample_bit_levels_vec(wave, Ns)
        # level changed? -> 1, same? -> 0
        prev_levels = np.r_[nrzi_start_level, levels[:-1]]
        bits = np.where(levels != prev_levels, 1, 0)
        return bits.tolist(), meta

    if scheme == "Manchester":
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _decode_manchester_vec(wave, Ns), meta

    if scheme in ("DiffManchester", "Differential Manchester"):
        Ns = ensure_even(Ns)
        meta["Ns_adjusted_even"] = Ns
        return _decode_diff_manchester_vec(wave, Ns, start_level=diff_start_level), meta

    if scheme in ("AMI", "Bipolar-AMI"):
        levels = _sample_bit_levels_vec(wave, Ns)
        # non-zero level -> 1, zero -> 0
        bits = np.where(levels != 0, 1, 0)
        return bits.tolist(), meta

    if scheme == "Pseudoternary":
        levels = _sample_bit_levels_vec(wave, Ns)
        # non-zero level -> 0, zero -> 1
        bits = np.where(levels != 0, 0, 1)
        return bits.tolist(), meta

    if scheme == "B8ZS":
        tern = _sample_bit_levels_vec(wave, Ns)
        bits, dmeta = _descramble_b8zs_opt(tern, last_pulse_init=last_pulse_init)
        meta.update(dmeta)
        return bits, meta

    if scheme == "HDB3":
        tern = _sample_bit_levels_vec(wave, Ns)
        bits, dmeta = _descramble_hdb3_opt(tern, last_nonzero_init=last_pulse_init)
        meta.update(dmeta)
        return bits, meta

    raise ValueError(f"Unknown scheme: {scheme}")


def simulate_d2d(
    bits: List[int],
    scheme: str,
    params: SimParams,
    *,
    nrzi_start_level: int = -1,
    diff_start_level: float = +1.0,
    last_pulse_init: int = -1,
    last_zero_pulse_init: int = -1,
    hdb3_nonzero_since_violation_init: int = 0,
) -> SimResult:
    tx, meta_tx = line_encode(
        bits, scheme, params,
        nrzi_start_level=nrzi_start_level,
        diff_start_level=diff_start_level,
        last_pulse_init=last_pulse_init,
        last_zero_pulse_init=last_zero_pulse_init,
        hdb3_nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
    )
    decoded, meta_rx = line_decode(
        tx, scheme, params,
        nrzi_start_level=nrzi_start_level,
        diff_start_level=diff_start_level,
        last_pulse_init=last_pulse_init,
        last_zero_pulse_init=last_zero_pulse_init,
    )

    n = len(tx)
    t = make_time_axis(n, params.fs)

    meta = {
        "scheme": scheme,
        "encode": meta_tx,
        "decode": meta_rx,
        "match": (decoded == bits),
        "input_len": len(bits),
    }
    return SimResult(
        t=t,
        signals={"tx": tx},
        bits={"input": bits, "decoded": decoded},
        meta=meta
    )