from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import base64

from utils import SimParams, bits_from_string, bits_to_string, gen_random_bits, bits_to_step
from d2d import simulate_d2d
from d2a import simulate_d2a
from a2d import simulate_a2d
from a2a import simulate_a2a

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        [data-testid="InputInstructions"] {
            display: none !important;
        }

        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"]{
            padding-top: 2.3rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            padding-bottom: 2.3rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def plot_signal(t, x, title, grid=False, step=False, x_dtick=1, y_dtick=1):
    fig = go.Figure()
    if step:
        fig.add_trace(go.Scatter(x=t, y=x, mode="lines", line_shape="hv", name=title))
    else:
        fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name=title))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    if grid:
        fig.update_xaxes(showgrid=True, tickmode="linear", dtick=x_dtick)
        fig.update_yaxes(showgrid=True, tickmode="linear", dtick=y_dtick)

    return fig

logo_path = Path(__file__).parent / "assets" / "itu-logo.png"

col_logo, col_text, col_ref = st.columns([1, 7, 4], vertical_alignment="center")
with col_logo:
    data = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <a href="https://itu.edu.tr/en/homepage" target="_blank" style="display:inline-block; line-height:0;">
          <img
            src="data:image/png;base64,{data}"
            alt="Istanbul Technical University"
            style="
              height:80px; width:auto;
              display:block;
              margin:0; padding:0;
              object-fit:contain;
            "
          />
        </a>
        """,
        unsafe_allow_html=True,
    )


with col_text:
    st.markdown(
        """
        <div style="font-size:0.98rem; opacity:0.85; line-height:1.35; margin-top:0.1rem;">
          This simulator was developed as a course assignment for
          <a href="https://ninova.itu.edu.tr/en/public.dersler.ders.aspx?dil=en&kategori=faculty-of-computer-and-informatics&dersId=5507"
             target="_blank"
             style="text-decoration:none; color:inherit;">
            <b>Principles of Computer Communications (BLG 337E)</b>
          </a><br/>
          <span style="opacity:0.75;">
            <a href="https://bbf.itu.edu.tr/en" target="_blank" style="text-decoration:none; color:inherit;">
              Computer Engineering Faculty
            </a>
            — 
            <a href="https://itu.edu.tr/en/homepage" target="_blank" style="text-decoration:none; color:inherit;">
              Istanbul Technical University
            </a>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_ref:
    st.markdown(
        """
        <div style="text-align:right; line-height:1.25; margin-top:0.15rem;">
            <div style="font-size:0.92rem; opacity:0.82;">
              All algorithms in this simulator follow
            </div>
            <a href="http://williamstallings.com/DataComm/" target="_blank" style="text-decoration:none; color:inherit;">
                <div style="font-size:0.86rem; opacity:0.72;">
                  <b>Data and Computer Communications (10th ed.)</b>
                  by William Stallings
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
        .creator-link {
          text-decoration: none;
          color: inherit;
          padding: 2px 6px;
          border-radius: 8px;
          transition: background-color 120ms ease, opacity 120ms ease;

          display: inline-flex;        /* NEW */
          align-items: center;         /* NEW: centers icon + text vertically */
          gap: 6px;                    /* NEW: space between icon and name */
        }

        .creator-link:hover {
          background-color: rgba(128, 128, 128, 0.18);
          opacity: 1.0;
        }

        .gh-icon {
          display: inline-flex;
          opacity: 0.85;
          transform: translateY(-1px); /* adjustable micro-shift; set to 0px if not needed */
        }

        .gh-icon svg {
          width: 16px;
          height: 16px;
          fill: currentColor;
          display: block;
        }
    </style>

    <div style="font-size:0.95rem; opacity:0.78; margin-bottom:0.5rem; margin-top:0.5rem; display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
      Created by:
      <a class="creator-link" href="https://github.com/itu-itis23-mahmoud21" target="_blank"><span class="gh-icon" aria-hidden="true">
        <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
          -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78
          -.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
          2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44
          1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54
          1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
        </svg>
      </span>Mohamed Ahmed Abdelsattar Mahmoud</a>
      <span>
        &amp;
      </span>
      <a class="creator-link" href="https://github.com/racha-badreddine" target="_blank"><span class="gh-icon" aria-hidden="true">
        <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
          -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78
          -.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
          2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44
          1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54
          1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
        </svg>
      </span>Racha Baddredine</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Encoding and Modulation Techniques Simulator")

with st.sidebar:
    st.header("Controls")

    mode = st.selectbox(
        "Mode",
        ["Digital → Digital", "Digital → Analog", "Analog → Digital", "Analog → Analog"],
    )

    show_grid = st.checkbox("Show grid", value=True)

    st.divider()

    # -------------------------
    # Common params (mode-aware)
    # -------------------------
    st.subheader("Common parameters")

    # Always meaningful (Digital sampling / time base)
    Ns = st.slider("Samples per bit (Ns)", 20, 400, 100, step=10)
    Tb = st.number_input("Bit duration Tb (s)", min_value=0.001, value=1.0, step=0.1)

    # Derived sampling rate
    fs = Ns / Tb
    st.text_input(
        "Sampling frequency\nfs = Ns / Tb  (Hz)",
        value=f"{fs:.6g}",
        disabled=True,
    )

    # Carrier controls only for Digital → Analog
    # (Keep hidden defaults for other modes so SimParams stays valid everywhere)
    Ac_default = 1.0
    cycles_default = 10

    if mode == "Digital → Analog":
        st.divider()
        st.subheader("Carrier parameters")

        Ac = st.number_input("Carrier amplitude Ac", min_value=0.1, value=1.0, step=0.1)
        cycles_per_bit = st.slider("Carrier cycles per bit", 2, 30, 10)
    else:
        Ac = Ac_default
        cycles_per_bit = cycles_default

    fc = float(cycles_per_bit) / float(Tb)

    # Show fc only when it matters (Digital → Analog)
    if mode == "Digital → Analog":
        st.text_input(
            "Carrier frequency\nfc = cycles_per_bit / Tb  (Hz)",
            value=f"{fc:.6g}",
            disabled=True,
        )

    params = SimParams(fs=fs, Tb=Tb, samples_per_bit=Ns, Ac=Ac, fc=fc)


    st.divider()

def summary_block(meta: dict):
    keys = ["scheme", "match", "input_len", "pad_info", "fs", "fc", "Tb", "samples_per_bit"]

    rows = []
    for k in keys:
        if k not in meta:
            continue
        v = meta[k]

        # Fixed formatting
        if k == "Tb" and isinstance(v, (int, float, np.floating)):
            v = f"{float(v):.2f}"
        elif k == "fs" and isinstance(v, (int, float, np.floating)):
            v = f"{float(v):.4f}"
        elif k == "fc" and isinstance(v, (int, float, np.floating)):
            v = f"{float(v):.4f}"

        rows.append({"Field": k, "Value": v})

    if not rows:
        return

    st.subheader("Summary")

    df = pd.DataFrame(rows)
    df = df.fillna("").astype(str)

    def _zebra(row):
        # shade every 2nd row lightly
        if row.name % 2 == 1:
            return ["background-color: rgba(128,128,128,0.10)"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(_zebra, axis=1)

    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=False,
        width=700, 
    )

def dict_to_pretty_table(data: dict, *, width: int = 700):
    """
    Render a dict as a 2-column zebra-striped table (like Summary).
    Falls back to showing non-scalars as compact strings.
    """
    if not isinstance(data, dict) or not data:
        st.info("No data.")
        return

    rows = []
    for k, v in data.items():
        # Keep simple scalars as-is
        if isinstance(v, (str, int, float, bool, type(None), np.floating)):
            rows.append({"Field": str(k), "Value": v})
        else:
            # For lists/dicts/arrays: show compact text (readable, not huge JSON)
            rows.append({"Field": str(k), "Value": str(v)})

    df = pd.DataFrame(rows)

    def _zebra(row):
        if row.name % 2 == 1:
            return ["background-color: rgba(128,128,128,0.10)"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(_zebra, axis=1)

    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=False,
        width=width,
    )

def render_events_table(events: list, *, width: int = 850):
    """Render a list[dict] as a zebra-striped table."""
    if not isinstance(events, list) or len(events) == 0:
        st.info("No events.")
        return

    # Make values readable (lists -> short strings)
    rows = []
    for e in events:
        if not isinstance(e, dict):
            rows.append({"event": str(e)})
            continue
        def _fmt_num(x):
            # Always return a STRING so Streamlit aligns everything the same way
            if isinstance(x, (np.integer, int)):
                return str(int(x))

            if isinstance(x, (np.floating, float)):
                xf = float(x)
                if np.isfinite(xf) and abs(xf - round(xf)) < 1e-9:
                    return str(int(round(xf)))
                return f"{xf:.4f}".rstrip("0").rstrip(".")

            if x is None:
                return ""

            return str(x)

        r = {}
        for k, v in e.items():
            if isinstance(v, list):
                # Format numeric lists nicely (patterns/chunks/windows)
                r[k] = "[" + ", ".join(str(_fmt_num(x)) for x in v) + "]"
            else:
                r[k] = _fmt_num(v)
        rows.append(r)

    df = pd.DataFrame(rows)

    def _zebra(row):
        if row.name % 2 == 1:
            return ["background-color: rgba(128,128,128,0.10)"] * len(row)
        return [""] * len(row)

    styler = (
        df.style
          .apply(_zebra, axis=1)
          .set_properties(**{"text-align": "left"})
          .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )

    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=False,
        width=width,
    )

def empty_state(message: str = "Click **Run simulation** from the sidebar to see results."):
    # Centered, friendly placeholder
    st.markdown(
        """
        <div style="text-align:center; padding: 6rem 1rem; opacity: 0.95;">
            <div style="font-size: 4rem; line-height: 1;">📡</div>
            <div style="font-size: 1.35rem; font-weight: 600; margin-top: 0.75rem;">
                Ready when you are
            </div>
            <div style="font-size: 1.05rem; margin-top: 0.5rem;">
        """
        + message +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def phase_as_pi_frac(x: float, *, max_den: int = 16) -> str:
    """
    Approximate x (radians) as k*pi/n with small denominator.
    Returns strings like 'π/4', '-3π/4', 'π', '0', etc.
    """
    if not np.isfinite(x):
        return ""

    # represent as multiple of pi
    r = float(x) / float(np.pi)

    best_num, best_den, best_err = 0, 1, float("inf")
    for den in range(1, max_den + 1):
        num = int(round(r * den))
        err = abs(r - (num / den))
        if err < best_err:
            best_num, best_den, best_err = num, den, err

    # build pretty string
    num = best_num
    den = best_den

    if num == 0:
        return "0"
    sign = "-" if num < 0 else ""
    num = abs(num)

    if den == 1:
        return f"{sign}π" if num == 1 else f"{sign}{num}π"
    else:
        if num == 1:
            return f"{sign}π/{den}"
        return f"{sign}{num}π/{den}"

def make_signature(mode: str, params: SimParams, **kwargs):
    # Use only JSON-serializable primitives
    base = {
        "mode": mode,
        "fs": float(params.fs),
        "Tb": float(params.Tb),
        "Ns": int(params.samples_per_bit),
        "Ac": float(params.Ac),
        "fc": float(params.fc),
    }
    base.update(kwargs)
    # Stable ordering
    return tuple(sorted(base.items()))

if mode in ("Digital → Digital", "Digital → Analog"):
    with st.sidebar:
        st.subheader("Digital input")
        default_bits = "10110010"

        # Applied value (used by simulation)
        if "bitstr" not in st.session_state:
            st.session_state["bitstr"] = default_bits

        # Draft value (what user edits)
        if "bitstr_draft" not in st.session_state:
            st.session_state["bitstr_draft"] = st.session_state["bitstr"]

        st.text_input("Bitstring (required)", key="bitstr_draft")

        # Validate bitstring and show the message under the input (sidebar)
        _tmp = st.session_state.get("bitstr_draft", "").strip()
        if _tmp == "":
            st.error("Bitstring is required.")
        else:
            bad = [ch for ch in _tmp if ch not in "01"]
            if bad:
                st.error("Bitstring must contain only 0 and 1.")

        st.slider("Random bits N", 8, 256, 32, step=8, key="rand_n")
        st.text_input("Seed (optional)", value="", key="rand_seed")

        # Validate seed (optional): must be integer if provided
        seed_txt = st.session_state.get("rand_seed", "").strip()
        seed_invalid = False
        if seed_txt != "":
            if seed_txt == "-" or not seed_txt.lstrip("-").isdigit():
                seed_invalid = True
                st.error("Seed must be an integer.")
        
        def _gen_bits_cb():
            seed_txt = st.session_state.get("rand_seed", "").strip()
            if seed_txt != "":
                if seed_txt == "-" or not seed_txt.lstrip("-").isdigit():
                    return  # invalid seed -> do nothing (sidebar error already shown)
                s = int(seed_txt)
            else:
                s = None
            n = int(st.session_state.get("rand_n", 32))
            st.session_state["bitstr_draft"] = bits_to_string(gen_random_bits(n, seed=s))

        st.button("Generate random bits", on_click=_gen_bits_cb)

    # Always read the current bitstring from session_state
    bitstr = st.session_state["bitstr"].strip()

    bitstr_draft = st.session_state.get("bitstr_draft", "").strip()
    draft_invalid = (bitstr_draft == "") or any(ch not in "01" for ch in bitstr_draft)

    bits = None
    bitstr_invalid = False

    try:
        bits = bits_from_string(bitstr)
    except Exception:
        bitstr_invalid = True

if mode == "Digital → Digital":
    with st.sidebar:
        st.subheader("Technique")
        line_amp = st.slider("Line amplitude (±A)", 1.0, 10.0, 1.0, step=0.5, key="d2d_line_amp")
        st.session_state.pop("d2d_scheme", None)

        base_scheme = st.selectbox(
            "Encoding Algorithm",
            ["NRZ-L", "NRZI", "Manchester", "Differential Manchester", "Bipolar-AMI", "Pseudoternary"],
            key="d2d_base_scheme",
        )

        # Scrambling selector only for AMI
        if base_scheme == "Bipolar-AMI":
            scramble = st.selectbox(
                "Scrambling Technique",
                ["None", "B8ZS", "HDB3"],
                key="d2d_scramble",
            )
            scheme = "Bipolar-AMI" if scramble == "None" else scramble
        else:
            scheme = base_scheme

        nrzi_start_level = -1
        diff_start_level = +1.0
        last_pulse_init = -1
        last_zero_pulse_init = -1

        # For NRZ-L / Manchester: initial display level before first bit (to show or not show a transition at t=0)
        nrzl_prev_level = +1          # +A means previous level was "high"
        manchester_prev_bit = 1       # assumed bit value immediately before the sequence starts (0 or 1)

        if scheme == "NRZ-L":
            opt = st.selectbox("Assumed level BEFORE first bit", ["High (+A)", "Low (-A)"], index=0, key="d2d_nrzl_prev")
            nrzl_prev_level = +1 if opt.startswith("High") else -1

        if scheme == "NRZI":
            opt = st.selectbox("Assumed level BEFORE first bit", ["Low (-A)", "High (+A)"], index=0, key="d2d_nrzi_prev")
            nrzi_start_level = -1 if opt.startswith("Low") else +1

        if scheme == "Manchester":
            opt = st.selectbox("Assumed preceding bit (affects transition at t=0)", ["1", "0"], index=0, key="d2d_manch_prevbit")
            manchester_prev_bit = 1 if opt == "1" else 0

        if scheme == "Differential Manchester":
            opt = st.selectbox("Assumed level BEFORE first bit", ["High (+A)", "Low (-A)"], index=0, key="d2d_diffman_prev")
            diff_start_level = +1.0 if opt.startswith("High") else -1.0

        if scheme == "Pseudoternary":
            opt = st.selectbox("Most recent preceding '0' polarity", ["Negative (-A)", "Positive (+A)"], index=0, key="d2d_pseudo_last0")
            last_zero_pulse_init = -1 if opt.startswith("Negative") else +1

        hdb3_nonzero_since_violation_init = 0  # 0=even, 1=odd

        if scheme in ("Bipolar-AMI", "B8ZS"):
            opt = st.selectbox("Most recent preceding '1' polarity", ["Negative (-A)", "Positive (+A)"], index=0, key="d2d_ami_last1")
            last_pulse_init = -1 if opt.startswith("Negative") else +1

        if scheme == "HDB3":
            opt1 = st.selectbox("Polarity of preceding pulse", ["Negative (-A)", "Positive (+A)"], index=0, key="d2d_hdb3_lastpulse")
            last_pulse_init = -1 if opt1.startswith("Negative") else +1

            opt2 = st.selectbox("Bipolar pulses since last substitution", ["Even", "Odd"], index=0, key="d2d_hdb3_parity")
            hdb3_nonzero_since_violation_init = 0 if opt2 == "Even" else 1

        current_sig = make_signature(
            "d2d", params,
            bitstr=bitstr,
            scheme=scheme,
            nrzi_start_level=int(nrzi_start_level),
            diff_start_level=float(diff_start_level),
            last_pulse_init=int(last_pulse_init),
            last_zero_pulse_init=int(last_zero_pulse_init),
            hdb3_nonzero_since_violation_init=int(hdb3_nonzero_since_violation_init),
        )

        compare_mode = st.checkbox("Compare mode (show multiple)", value=False, key="d2d_compare")
        run = st.button(
            "Run simulation",
            type="primary",
            key="d2d_run",
            disabled=(draft_invalid or seed_invalid),
        )

        # Auto-run if live and signature changed (or never ran before)
        if "d2d_last" not in st.session_state:
            st.session_state["d2d_last"] = None
        if "d2d_sig" not in st.session_state:
            st.session_state["d2d_sig"] = None

        live = True # always live for d2d

        prev_sig = st.session_state.get("d2d_sig", None)

        # If the user is editing a new bitstring draft (not yet applied), don't auto-rerun.
        draft_dirty = st.session_state.get("bitstr_draft", "").strip() != st.session_state.get("bitstr", "").strip()

        should_run = bool(run) or (live and (prev_sig is not None) and (prev_sig != current_sig) and (not draft_dirty))

        if should_run and (not seed_invalid):
            # Apply the draft bitstring ONLY when user clicks Run
            if run and (not draft_invalid):
                st.session_state["bitstr"] = st.session_state["bitstr_draft"].strip()

            # Use the applied bitstring for the actual simulation
            bitstr_run = st.session_state["bitstr"].strip()
            bits_run = bits_from_string(bitstr_run)

            # Recompute signature based on the applied bitstring
            current_sig = make_signature(
                "d2d", params,
                bitstr=bitstr_run,
                scheme=scheme,
                nrzi_start_level=int(nrzi_start_level),
                diff_start_level=float(diff_start_level),
                last_pulse_init=int(last_pulse_init),
                last_zero_pulse_init=int(last_zero_pulse_init),
                hdb3_nonzero_since_violation_init=int(hdb3_nonzero_since_violation_init),
            )

            st.session_state["d2d_last"] = simulate_d2d(
                bits_run, scheme, params,
                nrzi_start_level=nrzi_start_level,
                diff_start_level=diff_start_level,
                last_pulse_init=last_pulse_init,
                last_zero_pulse_init=last_zero_pulse_init,
                hdb3_nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
            )
            st.session_state["d2d_sig"] = current_sig

    res = st.session_state.get("d2d_last", None)

    if res is None:
        empty_state("Choose a scheme, optionally generate bits, then click **Run simulation**.")
    else:
        summary_block({**res.meta, "fs": params.fs, "Tb": params.Tb, "samples_per_bit": params.samples_per_bit})

        tab1, tab3, tab4 = st.tabs(["Waveforms", "Steps", "Details"])

        with tab1:
            tx_scaled = res.signals["tx"] * line_amp
            t_to_plot = res.t
            tx_to_plot = tx_scaled

            if scheme == "NRZ-L":
                init_level = nrzl_prev_level * line_amp
                t_to_plot = np.concatenate([[0.0], t_to_plot])
                tx_to_plot = np.concatenate([[init_level], tx_to_plot])

            if scheme == "NRZI":
                init_level = nrzi_start_level * line_amp
                t_to_plot = np.concatenate([[0.0], t_to_plot])          # duplicate t=0
                tx_to_plot = np.concatenate([[init_level], tx_to_plot]) # different y at same x => vertical line

            if scheme == "Manchester":
                prev_end_level = (+1 if manchester_prev_bit == 1 else -1)  # end level of preceding bit
                init_level = prev_end_level * line_amp
                t_to_plot = np.concatenate([[0.0], t_to_plot])
                tx_to_plot = np.concatenate([[init_level], tx_to_plot])
            
            if scheme == "Differential Manchester":
                init_level = diff_start_level * line_amp
                t_to_plot = np.concatenate([[0.0], t_to_plot])
                tx_to_plot = np.concatenate([[init_level], tx_to_plot])

            # Input bits as step
            inp = res.bits["input"]
            t_bits = np.arange(len(inp)*Ns) / params.fs
            x_bits = bits_to_step(inp, Ns)
            st.plotly_chart(plot_signal(t_bits, x_bits, "Input bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

            st.plotly_chart(plot_signal(t_to_plot, tx_to_plot, f"Encoded waveform ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')

            dec = res.bits["decoded"]
            t_dec = np.arange(len(dec)*Ns) / params.fs
            x_dec = bits_to_step(dec, Ns)
            st.plotly_chart(plot_signal(t_dec, x_dec, "Decoded bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

            if compare_mode:
                st.info("Compare mode: ON → showing all schemes with the same input bits.")
                cols = st.columns(2)

                def _prep_plot(s: str, t: np.ndarray, tx_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    # Apply the SAME “assumption visualization” rules used by the main Encoded waveform
                    t2 = t
                    x2 = tx_scaled

                    if s == "NRZ-L":
                        init_level = nrzl_prev_level * line_amp
                        t2 = np.concatenate([[0.0], t2])
                        x2 = np.concatenate([[init_level], x2])

                    if s == "NRZI":
                        init_level = nrzi_start_level * line_amp
                        t2 = np.concatenate([[0.0], t2])
                        x2 = np.concatenate([[init_level], x2])

                    if s == "Manchester":
                        prev_end_level = (+1 if manchester_prev_bit == 1 else -1)
                        init_level = prev_end_level * line_amp
                        t2 = np.concatenate([[0.0], t2])
                        x2 = np.concatenate([[init_level], x2])

                    if s == "Differential Manchester":
                        init_level = diff_start_level * line_amp
                        t2 = np.concatenate([[0.0], t2])
                        x2 = np.concatenate([[init_level], x2])

                    return t2, x2
                
                def _assumption_note_for_compare(s: str) -> str | None:
                    # Show notes only for plots that are NOT the currently selected scheme
                    if s == scheme:
                        return None

                    amp = f"{line_amp:g}"  # show numeric amplitude cleanly

                    if s == "NRZ-L":
                        lvl = "High" if nrzl_prev_level > 0 else "Low"
                        sign = "+" if nrzl_prev_level > 0 else "-"
                        return f"Assumed level BEFORE first bit: {lvl} ({sign}{amp})"

                    if s == "NRZI":
                        lvl = "High" if nrzi_start_level > 0 else "Low"
                        sign = "+" if nrzi_start_level > 0 else "-"
                        return f"Assumed level BEFORE first bit: {lvl} ({sign}{amp})"

                    if s == "Manchester":
                        return f"Assumed preceding bit: {manchester_prev_bit}"

                    if s == "Differential Manchester":
                        lvl = "High" if diff_start_level > 0 else "Low"
                        sign = "+" if diff_start_level > 0 else "-"
                        return f"Assumed level BEFORE first bit: {lvl} ({sign}{amp})"

                    if s == "Bipolar-AMI":
                        pol = "Positive" if last_pulse_init > 0 else "Negative"
                        sign = "+" if last_pulse_init > 0 else "-"
                        return f"Assumed most recent preceding '1' polarity: {pol} ({sign}{amp})"

                    if s == "Pseudoternary":
                        pol = "Positive" if last_zero_pulse_init > 0 else "Negative"
                        sign = "+" if last_zero_pulse_init > 0 else "-"
                        return f"Assumed most recent preceding '0' polarity: {pol} ({sign}{amp})"

                    return None

                for idx, s2 in enumerate(["NRZ-L", "NRZI", "Manchester", "Differential Manchester", "Bipolar-AMI", "Pseudoternary"]):
                    # IMPORTANT: pass the SAME init params so AMI/Pseudoternary/NRZI/DiffMan match the main plot
                    r2 = simulate_d2d(
                        res.bits["input"], s2, params,
                        nrzi_start_level=nrzi_start_level,
                        diff_start_level=diff_start_level,
                        last_pulse_init=last_pulse_init,
                        last_zero_pulse_init=last_zero_pulse_init,
                        hdb3_nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
                    )

                    tx2 = r2.signals["tx"] * line_amp
                    t2, tx2 = _prep_plot(s2, r2.t, tx2)

                    with cols[idx % 2]:
                        note = _assumption_note_for_compare(s2)
                        if note:
                            title = f"{s2}  <span style='font-size:0.80em; opacity:0.8'>'{note}'</span>"
                        else:
                            title = f"{s2}"

                        st.plotly_chart(
                            plot_signal(t2, tx2, title, grid=show_grid, x_dtick=params.Tb, y_dtick=1),
                            width="stretch",
                        )

        with tab3:
            def _scale_meta_for_display(meta_obj: dict, amp: float) -> dict:
                # Make a deep-ish copy and scale any ternary patterns (+/-1/0) to +/-amp/0 for display
                if not isinstance(meta_obj, dict):
                    return meta_obj
                m = dict(meta_obj)

                # scale substitutions patterns if present
                subs = m.get("substitutions", None)
                if isinstance(subs, list):
                    new_subs = []
                    for s in subs:
                        s2 = dict(s)
                        if "pattern" in s2 and isinstance(s2["pattern"], list):
                            s2["pattern"] = [float(v) * float(amp) for v in s2["pattern"]]
                        # if you stored last_pulse polarity, show scaled too (optional)
                        if "last_pulse" in s2 and isinstance(s2["last_pulse"], (int, float)):
                            s2["last_pulse"] = float(s2["last_pulse"]) * float(amp)
                        if "last_pulse_before" in s2 and isinstance(s2["last_pulse_before"], (int, float)):
                            s2["last_pulse_before"] = float(s2["last_pulse_before"]) * float(amp)
                        # make common numeric fields floats so Streamlit JSON colors them consistently
                        for k in ("pos", "count_since_last_sub"):
                            if k in s2 and isinstance(s2[k], (int, float)):
                                s2[k] = int(s2[k])
                        new_subs.append(s2)
                    m["substitutions"] = new_subs

                # scale decoder hits windows if present
                hits = m.get("descramble_hits", None)
                if isinstance(hits, list):
                    new_hits = []
                    for h in hits:
                        h2 = dict(h)
                        if "chunk" in h2 and isinstance(h2["chunk"], list):
                            h2["chunk"] = [float(v) * float(amp) for v in h2["chunk"]]
                        if "window" in h2 and isinstance(h2["window"], list):
                            h2["window"] = [float(v) * float(amp) for v in h2["window"]]
                        if "last_pulse" in h2 and isinstance(h2["last_pulse"], (int, float)):
                            h2["last_pulse"] = float(h2["last_pulse"]) * float(amp)
                        if "last_nonzero" in h2 and isinstance(h2["last_nonzero"], (int, float)):
                            h2["last_nonzero"] = float(h2["last_nonzero"]) * float(amp)
                        # make common numeric fields floats for consistent coloring
                        for k in ("pos", "v", "b"):
                            if k in h2 and isinstance(h2[k], (int, float)):
                                h2[k] = int(h2[k])

                        new_hits.append(h2)
                    m["descramble_hits"] = new_hits

                m["display_amplitude"] = f"{float(amp):.2f}"
                return m

            enc = _scale_meta_for_display(res.meta.get("encode", {}), line_amp)
            dec = _scale_meta_for_display(res.meta.get("decode", {}), line_amp)

            # Pull out the nested lists so they don't ruin the pretty table
            enc_subs = enc.pop("substitutions", None)
            dec_hits = dec.pop("descramble_hits", None)

            st.write("Encoder meta:")
            dict_to_pretty_table(enc, width=700)

            if isinstance(enc_subs, list):
                with st.expander("Encoder substitutions", expanded=True):
                    render_events_table(enc_subs, width=1000)

            st.write("Decoder meta:")
            dict_to_pretty_table(dec, width=700)

            if isinstance(dec_hits, list):
                with st.expander("Decoder descramble hits", expanded=True):
                    render_events_table(dec_hits, width=1000)

        with tab4:
            st.write("Input bits:")
            st.code(bits_to_string(res.bits["input"]))
            st.write("Decoded bits:")
            st.code(bits_to_string(res.bits["decoded"]))
            st.write("Match:")
            if res.meta["match"]:
                st.success("MATCH")
            else:
                st.error("MISMATCH")

elif mode == "Digital → Analog":
    with st.sidebar:
        st.subheader("Technique")
        label = st.selectbox("Modulation Technique", ["ASK", "BFSK", "MFSK", "BPSK", "DPSK", "QPSK", "QAM"])
        scheme = label

        st.subheader("Technique parameters")
        kwargs = {}
        invalid_params = False

        if scheme == "ASK":
            A0 = st.slider("A0", 0.0, 1.0, 0.0, step=0.05)
            A1 = st.slider("A1", 0.1, 2.0, 1.0, step=0.1)
            kwargs["A0"] = A0
            kwargs["A1"] = A1

            # Disallow ambiguous ASK (no amplitude difference)
            if np.isclose(A0, A1):
                st.error("Invalid ASK: A0 and A1 cannot be equal (no amplitude difference to detect).")
                invalid_params = True

        if scheme == "BFSK":
            nyq = float(params.fs) / 2.0
        
            # Max symmetric deviation so both tones stay within (0, Nyquist)
            dev_max = min(fc - 0.1, nyq - fc - 0.1)
        
            if dev_max <= 0:
                st.error("BFSK invalid: fc is too close to 0 or Nyquist for any symmetric tones.")
                invalid_params = True
                dev_max = 0.1  # avoid crashing the UI
        
            f0_min = float(fc - dev_max)
            f0_max = float(fc)
            f1_min = float(fc)
            f1_max = float(fc + dev_max)
        
            # Initialize state once
            if "bfsk_dev" not in st.session_state:
                st.session_state["bfsk_dev"] = float(min(2.0 / Tb, 0.8 * dev_max)) if dev_max > 0 else 0.5
        
            if "bfsk_f0" not in st.session_state or "bfsk_f1" not in st.session_state:
                d = float(st.session_state["bfsk_dev"])
                d = float(np.clip(d, 0.0, dev_max))
                st.session_state["bfsk_f0"] = float(fc - d)
                st.session_state["bfsk_f1"] = float(fc + d)
        
            # Clamp existing values to new ranges whenever fc/fs/Tb changes
            st.session_state["bfsk_f0"] = float(np.clip(float(st.session_state["bfsk_f0"]), f0_min, f0_max))
            st.session_state["bfsk_f1"] = float(np.clip(float(st.session_state["bfsk_f1"]), f1_min, f1_max))
        
            # Force symmetry around fc using the larger deviation of the two
            d0 = float(fc - st.session_state["bfsk_f0"])
            d1 = float(st.session_state["bfsk_f1"] - fc)
            d = float(np.clip(max(d0, d1), 0.0, dev_max))
            st.session_state["bfsk_dev"] = d
            st.session_state["bfsk_f0"] = float(fc - d)
            st.session_state["bfsk_f1"] = float(fc + d)
        
            def _sync_from_f0_slider():
                f0 = float(st.session_state["bfsk_f0"])
                d = float(np.clip(fc - f0, 0.0, dev_max))
                st.session_state["bfsk_dev"] = d
                st.session_state["bfsk_f0"] = float(fc - d)
                st.session_state["bfsk_f1"] = float(fc + d)
        
            def _sync_from_f1_slider():
                f1 = float(st.session_state["bfsk_f1"])
                d = float(np.clip(f1 - fc, 0.0, dev_max))
                st.session_state["bfsk_dev"] = d
                st.session_state["bfsk_f0"] = float(fc - d)
                st.session_state["bfsk_f1"] = float(fc + d)
        
            st.markdown(
                "<div style='font-size:0.85rem; opacity:0.75; margin-bottom:1rem;'>"
                "Book convention:<br/>f0 = fc − Δf and f1 = fc + Δf , |Δf| ≠ 0"
                "</div>",
                unsafe_allow_html=True,
            )

            st.slider(
                "f0 (Hz) for binary 0",
                min_value=f0_min,
                max_value=f0_max,
                step=0.1,
                key="bfsk_f0",
                on_change=_sync_from_f0_slider,
            )
        
            st.slider(
                "f1 (Hz) for binary 1",
                min_value=f1_min,
                max_value=f1_max,
                step=0.1,
                key="bfsk_f1",
                on_change=_sync_from_f1_slider,
            )
        
            st.text_input(
                "Derived deviation Δf (Hz)",
                value=f"{float(st.session_state['bfsk_dev']):.6g}",
                disabled=True,
            )
        
            kwargs["f0"] = float(st.session_state["bfsk_f0"])
            kwargs["f1"] = float(st.session_state["bfsk_f1"])
        
            if np.isclose(kwargs["f0"], kwargs["f1"]):
                st.error("Invalid BFSK: f0 and f1 cannot be equal.")
                invalid_params = True

        if scheme == "MFSK":
            nyq = float(params.fs) / 2.0

            L = st.select_slider("Bits per symbol (L)", options=[2, 3, 4], value=2)
            M = 2 ** int(L)
            st.text_input("Derived number of tones M = 2^L", value=str(M), disabled=True)

            # Keep all tones within (0, Nyquist): fc ± (M-1)*fd
            fd_max = min((fc - 0.1) / (M - 1), (nyq - fc - 0.1) / (M - 1))
            if fd_max <= 0:
                st.error("MFSK invalid: fc is too close to 0 or Nyquist for the selected M.")
                invalid_params = True
                fd_max = 0.1

            fd_default = min(1.0 / Tb, 0.5 * fd_max)

            fd = st.slider("Frequency difference fd (Hz)", 0.1, float(fd_max), float(fd_default), step=0.1)

            # Show the tone set
            freqs = [fc + (2 * (i + 1) - 1 - M) * fd for i in range(M)]
            with st.expander("MFSK tone frequencies"):
                rows = [{"tone_index": i, "f_i (Hz)": f"{float(f):.2f}"} for i, f in enumerate(freqs)]
                render_events_table(rows, width=700)

            kwargs["L"] = int(L)
            kwargs["fd"] = float(fd)

        if scheme == "BPSK":
            preset = st.selectbox(
                "BPSK configuration",
                ["Standard (0, π)", "Offset (φ, φ+π)", "Custom (φ1, φ0)"],
                index=0,
            )

            if preset == "Standard (0, π)":
                phase1 = 0.0
                phase0 = float(np.pi)

            elif preset == "Offset (φ, φ+π)":
                phi = st.slider(
                    "Reference phase φ (rad)",
                    float(-np.pi),
                    float(np.pi),
                    0.0,
                    step=0.01,
                )
                phase1 = float(phi)
                phase0 = float(((phi + np.pi) + np.pi) % (2 * np.pi) - np.pi)

            else:  # Custom
                phase1 = st.slider(
                    "Phase of binary 1, φ1 (rad)",
                    float(-np.pi),
                    float(np.pi),
                    0.0,
                    step=0.01,
                )
                phase0 = st.slider(
                    "Phase of binary 0, φ0 (rad)",
                    float(-np.pi),
                    float(np.pi),
                    float(np.pi),
                    step=0.01,
                )
            kwargs["phase1"] = float(phase1)
            kwargs["phase0"] = float(phase0)

            # Derived values table (always shown)
            dphi = (float(phase0) - float(phase1) + np.pi) % (2 * np.pi) - np.pi

            abs_sep = abs((float(phase0) - float(phase1) + np.pi) % (2*np.pi) - np.pi)

            if abs_sep < 0.10:  # choose threshold (e.g., 0.10 rad ≈ 5.7°)
                st.error("Invalid BPSK: φ0 and φ1 are too close → symbols become indistinguishable.")
                invalid_params = True

            rows = [
                {"Item": "Bit 1 phase φ1", "Value": f"{float(phase1):.2f} rad"},
                {"Item": "Bit 0 phase φ0", "Value": f"{float(phase0):.2f} rad"},
                {"Item": "Phase separation |Δφ| = |φ0-φ1|", "Value": f"{abs(float(dphi)):.2f} rad"},
                {"Item": "Antipodal? (|Δφ| = π)", "Value": str(bool(np.isclose(abs(dphi), np.pi, atol=1e-2)))},
            ]
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Item": st.column_config.TextColumn("Item", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                },
            )

        if scheme == "DPSK":
            phase_init = st.slider(
                "Reference phase (initial) φ_ref (rad)",
                float(-np.pi),
                float(np.pi),
                0.0,
                step=0.01,
            )

            kwargs["phase_init"] = float(phase_init)
            kwargs["delta_phase"] = float(np.pi)  # fixed per textbook DPSK

            rows = [
                {"Item": "Reference phase φ_ref", "Value": f"{float(phase_init):.2f} rad"},
                {"Item": "Bit-1 phase change Δφ", "Value": f"{float(np.pi):.2f} rad"},
                {"Item": "Bit-0 phase change", "Value": "0.00 rad"},
            ]
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Item": st.column_config.TextColumn("Item", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                },
            )
        
        if scheme == "QPSK":
            phi_ref = st.slider(
                "Constellation rotation φ_ref (rad)",
                float(-np.pi),
                float(np.pi),
                0.0,      # IMPORTANT: default 0 keeps book phases exactly (±π/4, ±3π/4)
                step=0.01,
            )
            kwargs["phi_ref"] = float(phi_ref)

            # Table: bits -> (I,Q) -> phase = φ_ref + atan2(Q,I)
            entries = [
                {"Bits": "11", "I": +1, "Q": +1},
                {"Bits": "01", "I": -1, "Q": +1},
                {"Bits": "00", "I": -1, "Q": -1},
                {"Bits": "10", "I": +1, "Q": -1},
            ]
            rows = []
            for e in entries:
                I = float(e["I"])
                Q = float(e["Q"])
                phase = float(phi_ref + np.arctan2(Q, I))
                # wrap to (-pi, pi] for nicer display
                phase = (phase + np.pi) % (2 * np.pi) - np.pi
                rows.append({
                    "Bits": e["Bits"],
                    "I level": f"{I:.0f}",
                    "Q level": f"{Q:.0f}",
                    "Phase": f"{phase:.2f} ≈ {phase_as_pi_frac(phase)} rad",
                })

            st.markdown(
                "<div style='font-size:0.85rem; opacity:0.75; margin-top:0.25rem;'>"
                "Book mapping (Eq. 5.7) is obtained when φ_ref = 0."
                "</div>",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Bits": st.column_config.TextColumn("Bits", width="small"),
                    "I level": st.column_config.TextColumn("I level", width="small"),
                    "Q level": st.column_config.TextColumn("Q level", width="small"),
                    "Phase": st.column_config.TextColumn("Phase", width="medium"),
                },
            )

        if scheme == "QAM":
            qam_variant = st.radio(
                "QAM variant",
                ["2-level ASK (QAM)", "4-level ASK (16-QAM)"],
                index=0,
                horizontal=True,
            )

            axis_levels = 2 if qam_variant.startswith("2-level") else 4
            kwargs["axis_levels"] = int(axis_levels)

            phi_ref = st.slider(
                "Constellation rotation φ_ref (rad)",
                float(-np.pi),
                float(np.pi),
                0.0,
                step=0.01,
            )
            kwargs["phi_ref"] = float(phi_ref)

            st.markdown(
                "<div style='font-size:0.85rem; opacity:0.75; margin-top:0.25rem;'>"
                "Book form: s(t) = d1(t) cos(2πfct) + d2(t) sin(2πfct). "
                "Input is split into I/Q streams by taking alternate bits."
                "</div>",
                unsafe_allow_html=True,
            )

            if axis_levels == 2:
                # 2-level: show 4 states (I,Q) in {-1,+1}
                entries = [
                    {"Bits (I,Q)": "00", "d1": -1, "d2": -1},
                    {"Bits (I,Q)": "01", "d1": -1, "d2": +1},
                    {"Bits (I,Q)": "10", "d1": +1, "d2": -1},
                    {"Bits (I,Q)": "11", "d1": +1, "d2": +1},
                ]
                rows = []
                for e in entries:
                    d1 = float(e["d1"])
                    d2 = float(e["d2"])
                    phase = float(phi_ref + np.arctan2(d2, d1))
                    phase = (phase + np.pi) % (2 * np.pi) - np.pi
                    rows.append({
                        "Bits (I,Q)": e["Bits (I,Q)"],
                        "d1": f"{d1:.0f}",
                        "d2": f"{d2:.0f}",
                        "Phase": f"{phase:.2f} ≈ {phase_as_pi_frac(phase)} rad",
                    })

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Bits (I,Q)": st.column_config.TextColumn("Bits (I,Q)", width="small"),
                        "d1": st.column_config.TextColumn("d1", width="small"),
                        "d2": st.column_config.TextColumn("d2", width="small"),
                        "Phase": st.column_config.TextColumn("Phase", width="medium"),
                    },
                )

            else:
                # 16-QAM: show per-axis Gray mapping table (2 bits -> level)
                axis_rows = [
                    {"Axis bits": "00", "Level": "-3"},
                    {"Axis bits": "01", "Level": "-1"},
                    {"Axis bits": "11", "Level": "+1"},
                    {"Axis bits": "10", "Level": "+3"},
                ]
                st.markdown(
                    "<div style='font-size:0.85rem; opacity:0.75; margin-top:0.25rem;'>"
                    "16-QAM uses 4 levels per axis (Gray). One symbol uses 4 input bits in book order: "
                    "[I0, Q0, I1, Q1]. (I stream = 1st,3rd,5th,... bits; Q stream = 2nd,4th,6th,... bits.)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                df = pd.DataFrame(axis_rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Axis bits": st.column_config.TextColumn("Axis bits", width="small"),
                        "Level": st.column_config.TextColumn("Level", width="small"),
                    },
                )

        current_sig = make_signature(
            "d2a", params,
            bitstr=st.session_state["bitstr"].strip(),
            scheme=scheme,
            kwargs=tuple(sorted((k, float(v)) for k, v in kwargs.items())),
        )

        run = st.button(
            "Run simulation",
            type="primary",
            disabled=(invalid_params or draft_invalid or seed_invalid),
        )

    # --- D2A run state (aligned with D2D) ---
    if "d2a_last" not in st.session_state:
        st.session_state["d2a_last"] = None
    if "d2a_sig" not in st.session_state:
        st.session_state["d2a_sig"] = None
    
    live = True  # same idea as D2D: auto-update after first run when params change
    prev_sig = st.session_state.get("d2a_sig", None)
    
    # Don't auto-rerun while user is typing a new (not applied) bitstring draft
    draft_dirty = st.session_state.get("bitstr_draft", "").strip() != st.session_state.get("bitstr", "").strip()
    
    should_run = bool(run) or (live and (prev_sig is not None) and (prev_sig != current_sig) and (not draft_dirty))
    
    if should_run and (not seed_invalid) and (not invalid_params):
        # Apply draft -> applied ONLY when user clicks Run
        if run and (not draft_invalid):
            st.session_state["bitstr"] = st.session_state["bitstr_draft"].strip()
    
        bitstr_run = st.session_state["bitstr"].strip()
        bits_run = bits_from_string(bitstr_run)
    
        # Recompute sig based on what we actually ran
        current_sig = make_signature(
            "d2a", params,
            bitstr=bitstr_run,
            scheme=scheme,
            kwargs=tuple(sorted((k, float(v)) for k, v in kwargs.items())),
        )
    
        st.session_state["d2a_last"] = simulate_d2a(bits_run, scheme, params, **kwargs)
        st.session_state["d2a_sig"] = current_sig
    
    res = st.session_state.get("d2a_last", None)
    
    if res is None:
        empty_state("Pick a modulation, then click **Run simulation** to generate the waveforms.")
    else:
        summary_block({**res.meta, "fs": params.fs, "fc": params.fc, "Tb": params.Tb, "samples_per_bit": params.samples_per_bit})

        tab1, tab3, tab4 = st.tabs(["Waveforms", "Steps", "Details"])

        with tab1:
            inp = res.bits["input"]
            t_bits = np.arange(len(inp)*Ns) / params.fs
            x_bits = bits_to_step(inp, Ns)
            st.plotly_chart(plot_signal(t_bits, x_bits, "Input bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["tx"], f"Modulated signal ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')

            dec = res.bits["decoded"]
            t_dec = np.arange(len(dec)*Ns) / params.fs
            x_dec = bits_to_step(dec, Ns)
            st.plotly_chart(plot_signal(t_dec, x_dec, "Recovered bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

        with tab3:
            dem = res.meta.get("demodulate", {})

            if isinstance(dem, dict):
                # 1) Make a copy so we can format without touching the original
                dem2 = dict(dem)

                # 2) ASK-specific formatting (safe even if the key isn't present)
                if "A0" in dem2 and isinstance(dem2["A0"], (int, float, np.floating)):
                    dem2["A0"] = f"{float(dem2['A0']):.2f}"
                if "A1" in dem2 and isinstance(dem2["A1"], (int, float, np.floating)):
                    dem2["A1"] = f"{float(dem2['A1']):.2f}"
                if "thr" in dem2 and isinstance(dem2["thr"], (int, float, np.floating)):
                    dem2["thr"] = f"{float(dem2['thr']):.3f}"  # good balance for threshold

                # Pull A_hat out so it doesn't appear as one huge list in the main table
                a_hat = dem2.pop("A_hat", None)
                i_hat = dem2.pop("I_hat", None)
                q_hat = dem2.pop("Q_hat", None)
                i_dec = dem2.pop("I_dec", None)
                q_dec = dem2.pop("Q_dec", None)
                phi_hat = dem2.pop("phi_hat", None)
                delta_hat = dem2.pop("delta_hat", None)
                warnings_list = dem2.pop("warnings", None)
                tone_sep = dem2.pop("tone_sep", None)
                dem2.pop("E0", None)
                dem2.pop("E1", None)

                # MFSK: format fd to 2 decimals
                if dem2.get("scheme") == "MFSK" and "fd" in dem2 and isinstance(dem2["fd"], (int, float, np.floating)):
                    dem2["fd"] = f"{float(dem2['fd']):.2f}"

                # MFSK: remove heavy lists from the main Steps table (keep internally in meta)
                mfsk_freqs = None
                mfsk_chosen = None
                if dem2.get("scheme") == "MFSK":
                    mfsk_freqs = dem2.pop("freqs", None)
                    mfsk_chosen = dem2.pop("chosen_idx", None)
                    dem2.pop("energies", None)  # hide from Steps tab only

                # BFSK-specific formatting
                if dem2.get("scheme") == "BFSK":
                    for fk in ("f0", "f1"):
                        if fk in dem2 and isinstance(dem2[fk], (int, float, np.floating)):
                            dem2[fk] = f"{float(dem2[fk]):.2f}"

                if dem2.get("scheme") == "BPSK":
                    for pk in ("phase1", "phase0"):
                        if pk in dem2 and isinstance(dem2[pk], (int, float, np.floating)):
                            dem2[pk] = f"{float(dem2[pk]):.2f}"
                    
                if dem2.get("scheme") == "DPSK":
                    for pk in ("phase_init", "delta_phase"):
                        if pk in dem2 and isinstance(dem2[pk], (int, float, np.floating)):
                            dem2[pk] = f"{float(dem2[pk]):.2f}"
                
                if dem2.get("scheme") == "QPSK":
                    if "phi_ref" in dem2 and isinstance(dem2["phi_ref"], (int, float, np.floating)):
                        dem2["phi_ref"] = f"{float(dem2['phi_ref']):.2f}"

                if dem2.get("scheme") == "QAM":
                    # show clean ints
                    for ik in ("axis_levels", "bits_per_axis", "bits_per_symbol", "symbols"):
                        if ik in dem2:
                            try:
                                dem2[ik] = int(dem2[ik])
                            except Exception:
                                pass
                            
                    # show clean floats
                    for fk in ("phi_ref", "norm"):
                        if fk in dem2 and isinstance(dem2[fk], (int, float, np.floating)):
                            dem2[fk] = f"{float(dem2[fk]):.2f}"

                # 3) Keep your existing “simple vs event list” logic
                simple = {}
                event_lists = {}
                for k, v in dem2.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        event_lists[k] = v
                    else:
                        simple[k] = v

                dict_to_pretty_table(simple, width=700)

                # MFSK: show freqs and chosen_idx as separate tables
                if isinstance(mfsk_freqs, list):
                    with st.expander("freqs (tone set)", expanded=True):
                        rows = [{"tone_index": i, "f_i (Hz)": f"{float(f):.2f}"} for i, f in enumerate(mfsk_freqs)]
                        render_events_table(rows, width=900)

                if isinstance(mfsk_chosen, list):
                    with st.expander("chosen_idx (detected tone index per symbol)", expanded=True):
                        rows = [{"symbol_index": i, "chosen_idx": int(idx)} for i, idx in enumerate(mfsk_chosen)]
                        render_events_table(rows, width=900)

                # --- Combined I/Q tables: hat vs dec (only when both exist) ---
                bpsym = None
                try:
                    bpsym = int(dem.get("bits_per_symbol"))
                except Exception:
                    # Fallbacks for older schemes that may not store bits_per_symbol explicitly
                    if dem2.get("scheme") == "16QAM":
                        bpsym = 4
                    elif dem2.get("scheme") == "QPSK":
                        bpsym = 2
                    else:
                        bpsym = 1  # default: 1 estimate per bit

                same_index = (bpsym == 1)

                def _make_index_cols(sym_i: int) -> dict:
                    if same_index:
                        return {"bit_index / symbol_index": sym_i}
                    # show both when they differ
                    return {
                        "symbol_index": sym_i,
                        "bit_index_start": sym_i * bpsym,
                    }

                # I table
                if isinstance(i_hat, list) and isinstance(i_dec, list):
                    with st.expander("I: I_hat vs I_dec", expanded=True):
                        n = min(len(i_hat), len(i_dec))
                        rows = []
                        for k in range(n):
                            r = _make_index_cols(k)
                            r["I_hat"] = f"{float(i_hat[k]):.2f}"
                            r["I_dec"] = f"{float(i_dec[k]):.2f}"
                            rows.append(r)
                        render_events_table(rows, width=1000)

                # Q table
                if isinstance(q_hat, list) and isinstance(q_dec, list):
                    with st.expander("Q: Q_hat vs Q_dec", expanded=True):
                        n = min(len(q_hat), len(q_dec))
                        rows = []
                        for k in range(n):
                            r = _make_index_cols(k)
                            r["Q_hat"] = f"{float(q_hat[k]):.2f}"
                            r["Q_dec"] = f"{float(q_dec[k]):.2f}"
                            rows.append(r)
                        render_events_table(rows, width=1000)

                # For schemes that only have hat (no dec), keep a simple display
                if isinstance(i_hat, list) and not isinstance(i_dec, list):
                    with st.expander("I_hat", expanded=True):
                        rows = [{"bit_index / symbol_index": i, "I_hat": f"{float(v):.2f}"} for i, v in enumerate(i_hat)]
                        render_events_table(rows, width=1000)

                if isinstance(q_hat, list) and not isinstance(q_dec, list):
                    with st.expander("Q_hat", expanded=True):
                        rows = [{"bit_index / symbol_index": i, "Q_hat": f"{float(v):.2f}"} for i, v in enumerate(q_hat)]
                        render_events_table(rows, width=1000)
                
                if isinstance(phi_hat, list):
                    with st.expander("phi_hat (estimated absolute phase per bit)", expanded=False):
                        rows = [{"bit_index": i, "phi_hat (rad)": f"{float(v):.2f}"} for i, v in enumerate(phi_hat)]
                        render_events_table(rows, width=1000)

                if isinstance(delta_hat, list):
                    with st.expander("delta_hat (estimated phase change per bit)", expanded=False):
                        rows = [{"bit_index": i, "delta_hat (rad)": f"{float(v):.2f}"} for i, v in enumerate(delta_hat)]
                        render_events_table(rows, width=1000)

                if isinstance(warnings_list, list) and len(warnings_list) > 0:
                    with st.expander("Warnings", expanded=True):
                        wrows = [{"warning": w} for w in warnings_list]
                        render_events_table(wrows, width=1000)

                # 5) Existing event tables (if any)
                for name, ev in event_lists.items():
                    with st.expander(name, expanded=True):
                        render_events_table(ev, width=1000)
            else:
                st.write(dem)
            
        with tab4:
            st.write("Input bits:")
            st.code(bits_to_string(res.bits["input"]))

            st.write("Decoded bits:")
            st.code(bits_to_string(res.bits["decoded"]))

            st.write("Match:")
            if res.meta["match"]:
                st.success("MATCH")
            else:
                st.error("MISMATCH")

elif mode == "Analog → Digital":
    with st.sidebar:
        st.subheader("Technique")
        technique = st.selectbox("Digitization", ["PCM", "DM"])

        st.subheader("Message signal")
        kind = st.selectbox("Waveform", ["sine", "square", "triangle"])
        Am = st.slider("Amplitude Am", 0.1, 5.0, 1.0, step=0.1)
        fm = st.slider("Message frequency fm (Hz)", 1.0, 50.0, 5.0, step=1.0)
        duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, step=0.5)

        st.subheader("Sampling")
        fs_mult = st.select_slider("Sampling multiplier (×fm)", options=[4, 8, 16, 32], value=8)

        st.subheader("Technique parameters")
        pcm_nbits = 4
        dm_delta = 0.1
        if technique == "PCM":
            pcm_nbits = st.select_slider("PCM bits per sample", options=[2, 3, 4, 5, 6], value=4)
        else:
            dm_delta = st.slider("DM step size Δ", 0.01, 1.0, 0.1, step=0.01)

        st.subheader("Line coding for produced bitstream")
        linecode_scheme = st.selectbox("Line code", ["NRZ-L", "Manchester", "NRZI", "Bipolar-AMI"])

        run = st.button("Run simulation", type="primary")

    if run:
        res = simulate_a2d(
            kind, technique, params,
            Am=Am, fm=fm, duration=duration,
            fs_mult=int(fs_mult),
            pcm_nbits=int(pcm_nbits),
            dm_delta=float(dm_delta),
            linecode_scheme=linecode_scheme
        )

        tab1, tab3, tab4 = st.tabs(["Waveforms", "Steps", "Details"])

        with tab1:
            st.plotly_chart(plot_signal(res.t, res.signals["m(t)"], "Message m(t)", grid=show_grid), width='stretch')

            # sampled points
            t_s = res.meta["sampled"]["t_s"]
            m_s = res.meta["sampled"]["m_s"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.t, y=res.signals["m(t)"], mode="lines", name="m(t)"))
            fig.add_trace(go.Scatter(x=t_s, y=m_s, mode="markers", name="Samples"))
            fig.update_layout(title="Message with sampled points", xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig, width='stretch')

            if technique == "PCM":
                q = res.meta["quantized"]["q"]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t_s, y=q, mode="lines", line_shape="hv", name="Quantized (stair)"))
                fig2.update_layout(title="PCM Quantized Staircase", xaxis_title="Time (s)", yaxis_title="Amplitude")
                st.plotly_chart(fig2, width='stretch')
            else:
                stair = res.meta["stair"]["stair"]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t_s, y=stair, mode="lines", line_shape="hv", name="DM Staircase"))
                fig2.update_layout(title="Delta Modulation Staircase", xaxis_title="Time (s)", yaxis_title="Amplitude")
                st.plotly_chart(fig2, width='stretch')

            t_bits = res.meta["t_bits"]
            st.plotly_chart(
                plot_signal(t_bits, res.signals["linecode"], f"Line-coded bitstream ({linecode_scheme})",
                            grid=show_grid, x_dtick=params.Tb, y_dtick=1),
                width='stretch'
            )

        with tab3:
            st.json(res.meta)

        with tab4:
            bits = res.bits["bitstream"]
            s = "".join("1" if b else "0" for b in bits)
            st.write(f"Bitstream length: {len(bits)}")
            st.code(s[:500] + ("..." if len(s) > 500 else ""))


elif mode == "Analog → Analog":
    with st.sidebar:
        st.subheader("Technique")
        scheme = st.selectbox("Modulation", ["AM", "FM", "PM"])

        st.subheader("Message signal")
        kind = st.selectbox("Waveform", ["sine", "square", "triangle"])
        Am = st.slider("Amplitude Am", 0.1, 5.0, 1.0, step=0.1)
        fm = st.slider("Message frequency fm (Hz)", 1.0, 50.0, 5.0, step=1.0)
        duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, step=0.5)

        st.subheader("Carrier/sampling")
        # For analog modes, we want a higher fs than digital fs:
        fs_a = st.select_slider("Sampling rate fs (Hz)", options=[2000, 5000, 10000, 20000], value=10000)
        fc_ratio = st.slider("Carrier ratio fc/fm", 5, 50, 20)
        fc_a = fc_ratio * fm
        Ac_a = st.number_input("Carrier amplitude Ac", min_value=0.1, value=1.0, step=0.1)

        st.subheader("Modulation parameter")
        ka = 0.5
        kf = 5.0
        kp = 1.0
        if scheme == "AM":
            ka = st.slider("ka", 0.0, 2.0, 0.5, step=0.05)
        elif scheme == "FM":
            kf = st.slider("kf", 0.1, 20.0, 5.0, step=0.5)
        else:
            kp = st.slider("kp", 0.1, 10.0, 1.0, step=0.1)

        run = st.button("Run simulation", type="primary")

    if run:
        aparams = SimParams(fs=float(fs_a), Tb=params.Tb, samples_per_bit=params.samples_per_bit, Ac=float(Ac_a), fc=float(fc_a))
        res = simulate_a2a(kind, scheme, aparams, Am=Am, fm=fm, duration=duration, ka=ka, kf=kf, kp=kp)

        tab1, tab3, tab4 = st.tabs(["Waveforms", "Steps", "Details"])

        with tab1:
            st.plotly_chart(plot_signal(res.t, res.signals["m(t)"], "Message m(t)", grid=show_grid), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["tx"], f"Modulated signal ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["recovered"], "Recovered message", grid=show_grid), width='stretch')

        with tab3:
            st.json(res.meta)

        with tab4:
            st.write("Parameters:")
            st.json(res.meta)
