from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import SimParams, bits_from_string, bits_to_string, gen_random_bits, bits_to_step, fft_mag
from d2d import simulate_d2d
from d2a import simulate_d2a
from a2d import simulate_a2d
from a2a import simulate_a2a

st.set_page_config(layout="wide")

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


def plot_freq(x, fs, title):
    f, mag = fft_mag(np.asarray(x, dtype=float), fs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=mag, mode="lines", name="|X(f)|"))
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    return fig


st.title("Principles of Computer Communications — Transmission Simulator")

with st.sidebar:
    st.header("Controls")

    mode = st.selectbox(
        "Mode",
        ["Digital → Digital", "Digital → Analog", "Analog → Digital", "Analog → Analog"],
    )

    show_grid = st.checkbox("Show grid", value=True)

    st.divider()

    # Common params
    st.subheader("Common parameters")
    Ns = st.slider("Samples per bit (Ns)", 20, 400, 100, step=10)
    Tb = st.number_input("Bit duration Tb (s)", min_value=0.001, value=1.0, step=0.1)

    # For analog carriers / analog modes
    Ac = st.number_input("Carrier amplitude Ac", min_value=0.1, value=1.0, step=0.1)
    cycles_per_bit = st.slider("Carrier cycles per bit (for passband)", 2, 30, 10)
    fc = float(cycles_per_bit) / float(Tb)

    # Non-editable derived value (updates automatically when Tb or cycles_per_bit changes)
    st.text_input(
        "Carrier frequency\nfc = cycles_per_bit / Tb  (Hz)",
        value=f"{fc:.6g}",
        disabled=True,
    )

    # Default fs based on digital sampling; analog modes override if needed
    fs = Ns / Tb

    params = SimParams(fs=fs, Tb=Tb, samples_per_bit=Ns, Ac=Ac, fc=fc)

    st.divider()


def summary_block(meta: dict):
    items = []
    for k in ["scheme", "match", "input_len", "pad_info", "fs", "fc", "Tb", "samples_per_bit"]:
        if k in meta:
            items.append((k, meta[k]))
    if items:
        st.subheader("Summary")
        st.json({k: v for k, v in items})

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


if mode in ("Digital → Digital", "Digital → Analog"):
    with st.sidebar:
        st.subheader("Digital input")
        default_bits = "10110010"

        # Initialize the widget state once
        if "bitstr" not in st.session_state:
            st.session_state["bitstr"] = default_bits

        st.text_input("Bitstring", key="bitstr")

        st.slider("Random bits N", 8, 256, 32, step=8, key="rand_n")
        st.text_input("Seed (optional)", value="", key="rand_seed")

        def _gen_bits_cb():
            seed_txt = st.session_state.get("rand_seed", "").strip()
            s = int(seed_txt) if seed_txt else None
            n = int(st.session_state.get("rand_n", 32))
            st.session_state["bitstr"] = bits_to_string(gen_random_bits(n, seed=s))

        st.button("Generate random bits", on_click=_gen_bits_cb)

    # Always read the current bitstring from session_state
    bitstr = st.session_state["bitstr"]

    try:
        bits = bits_from_string(bitstr)
    except Exception as e:
        st.error(str(e))
        st.stop()

if mode == "Digital → Digital":
    with st.sidebar:
        st.subheader("Technique")
        line_amp = st.slider("Line amplitude (±A)", 1.0, 10.0, 1.0, step=0.5)
        scheme = st.selectbox(
            "Line coding / Scrambling",
            ["NRZ-L", "NRZI", "Manchester", "Differential Manchester", "Bipolar-AMI", "Pseudoternary", "B8ZS", "HDB3"]
        )

        nrzi_start_level = -1
        diff_start_level = +1.0
        last_pulse_init = -1
        last_zero_pulse_init = -1

        # For NRZ-L / Manchester: initial display level before first bit (to show or not show a transition at t=0)
        nrzl_prev_level = +1          # +A means previous level was "high"
        manchester_prev_bit = 1       # assumed bit value immediately before the sequence starts (0 or 1)

        if scheme == "NRZ-L":
            opt = st.selectbox("Assumed level BEFORE first bit", ["High (+A)", "Low (-A)"], index=0)
            nrzl_prev_level = +1 if opt.startswith("High") else -1

        if scheme == "NRZI":
            opt = st.selectbox("Assumed level BEFORE first bit", ["Low (-A)", "High (+A)"], index=0)
            nrzi_start_level = -1 if opt.startswith("Low") else +1

        if scheme == "Manchester":
            opt = st.selectbox("Assumed preceding bit (affects transition at t=0)", ["1", "0"], index=0)
            manchester_prev_bit = 1 if opt == "1" else 0

        if scheme == "Differential Manchester":
            opt = st.selectbox("Assumed level BEFORE first bit", ["High (+A)", "Low (-A)"], index=0)
            diff_start_level = +1.0 if opt.startswith("High") else -1.0

        if scheme == "Pseudoternary":
            opt = st.selectbox("Most recent preceding '0' polarity", ["Negative (-A)", "Positive (+A)"], index=0)
            last_zero_pulse_init = -1 if opt.startswith("Negative") else +1

        hdb3_nonzero_since_violation_init = 0  # 0=even, 1=odd

        if scheme in ("Bipolar-AMI", "B8ZS"):
            opt = st.selectbox("Most recent preceding '1' polarity", ["Negative (-A)", "Positive (+A)"], index=0)
            last_pulse_init = -1 if opt.startswith("Negative") else +1

        if scheme == "HDB3":
            opt1 = st.selectbox("Polarity of preceding pulse", ["Negative (-A)", "Positive (+A)"], index=0)
            last_pulse_init = -1 if opt1.startswith("Negative") else +1

            opt2 = st.selectbox("Bipolar pulses since last substitution", ["Even", "Odd"], index=0)
            hdb3_nonzero_since_violation_init = 0 if opt2 == "Even" else 1

        compare_mode = st.checkbox("Compare mode (show multiple)", value=False)
        run = st.button("Run simulation", type="primary")

    if "d2d_last" not in st.session_state:
        st.session_state["d2d_last"] = None

    if run:
        st.session_state["d2d_last"] = simulate_d2d(
            bits, scheme, params,
            nrzi_start_level=nrzi_start_level,
            diff_start_level=diff_start_level,
            last_pulse_init=last_pulse_init,
            last_zero_pulse_init=last_zero_pulse_init,
            hdb3_nonzero_since_violation_init=hdb3_nonzero_since_violation_init,
        )
    
    res = st.session_state.get("d2d_last", None)
    if res is None:
        empty_state("Choose a scheme, optionally generate bits, then click **Run simulation**.")
    else:
        st.subheader("Results")
        summary_block({**res.meta, "fs": params.fs, "fc": params.fc, "Tb": params.Tb, "samples_per_bit": params.samples_per_bit})

        tab1, tab2, tab3, tab4 = st.tabs(["Waveforms", "Frequency", "Steps", "Details"])

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
            t_bits = np.arange(len(bits)*Ns) / params.fs
            x_bits = bits_to_step(bits, Ns)
            st.plotly_chart(plot_signal(t_bits, x_bits, "Input bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

            st.plotly_chart(plot_signal(t_to_plot, tx_to_plot, f"Encoded waveform ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')

            dec = res.bits["decoded"]
            t_dec = np.arange(len(dec)*Ns) / params.fs
            x_dec = bits_to_step(dec, Ns)
            st.plotly_chart(plot_signal(t_dec, x_dec, "Decoded bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

            if compare_mode:
                st.info("Compare mode: showing a few key schemes for the same input.")
                cols = st.columns(2)
                for idx, s2 in enumerate(["NRZ-L", "NRZI", "Manchester", "Bipolar-AMI"]):
                    r2 = simulate_d2d(bits, s2, params)
                    with cols[idx % 2]:
                        st.plotly_chart(plot_signal(r2.t, r2.signals["tx"] * line_amp, f"{s2}", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')

        with tab2:
            st.plotly_chart(plot_freq(res.signals["tx"], params.fs, "Spectrum of encoded waveform"), width='stretch')

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
                                s2[k] = float(s2[k])
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
                                h2[k] = float(h2[k])
                        new_hits.append(h2)
                    m["descramble_hits"] = new_hits

                m["display_amplitude"] = amp
                return m

            st.write("Encoder meta:")
            st.json(_scale_meta_for_display(res.meta.get("encode", {}), line_amp))

            st.write("Decoder meta:")
            st.json(_scale_meta_for_display(res.meta.get("decode", {}), line_amp))


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
        scheme = st.selectbox("Modulation", ["ASK", "BFSK", "BPSK", "QPSK", "16QAM"])

        st.subheader("Technique parameters")
        kwargs = {}
        if scheme == "ASK":
            kwargs["A0"] = st.slider("A0", 0.0, 1.0, 0.0, step=0.05)
            kwargs["A1"] = st.slider("A1", 0.1, 2.0, 1.0, step=0.1)
        if scheme == "BFSK":
            kwargs["tone_sep"] = st.slider("Tone separation (in 1/Tb units)", 0.5, 6.0, 2.0, step=0.5)

        run = st.button("Run simulation", type="primary")

    if "d2a_last" not in st.session_state:
        st.session_state["d2a_last"] = None

    if run:
        st.session_state["d2a_last"] = simulate_d2a(bits, scheme, params, **kwargs)
    res = st.session_state.get("d2a_last", None)
    
    if res is None:
        empty_state("Pick a modulation, then click **Run simulation** to generate the waveforms.")
    else:
        st.subheader("Results")
        summary_block({**res.meta, "fs": params.fs, "fc": params.fc, "Tb": params.Tb, "samples_per_bit": params.samples_per_bit})

        tab1, tab2, tab3, tab4 = st.tabs(["Waveforms", "Frequency", "Steps", "Details"])

        with tab1:
            t_bits = np.arange(len(bits)*Ns) / params.fs
            x_bits = bits_to_step(bits, Ns)
            st.plotly_chart(plot_signal(t_bits, x_bits, "Input bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["tx"], f"Modulated signal ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')

            dec = res.bits["decoded"]
            t_dec = np.arange(len(dec)*Ns) / params.fs
            x_dec = bits_to_step(dec, Ns)
            st.plotly_chart(plot_signal(t_dec, x_dec, "Recovered bits (0/1)", grid=show_grid, step=True, x_dtick=params.Tb, y_dtick=1), width='stretch')

        with tab2:
            st.plotly_chart(plot_freq(res.signals["tx"], params.fs, "Spectrum of modulated signal"), width='stretch')

        with tab3:
            st.json(res.meta.get("demodulate", {}))

        with tab4:
            st.code("Input:   " + bits_to_string(bits))
            st.code("Decoded:  " + bits_to_string(res.bits["decoded"]))
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

        st.subheader("Results")
        tab1, tab2, tab3, tab4 = st.tabs(["Waveforms", "Frequency", "Steps", "Details"])

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


        with tab2:
            st.plotly_chart(plot_freq(res.signals["m(t)"], res.meta["fs_display"], "Spectrum of message"), width='stretch')

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

        tab1, tab2, tab3, tab4 = st.tabs(["Waveforms", "Frequency", "Steps", "Details"])

        with tab1:
            st.plotly_chart(plot_signal(res.t, res.signals["m(t)"], "Message m(t)", grid=show_grid), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["tx"], f"Modulated signal ({scheme})", grid=show_grid, x_dtick=params.Tb, y_dtick=1), width='stretch')
            st.plotly_chart(plot_signal(res.t, res.signals["recovered"], "Recovered message", grid=show_grid), width='stretch')

        with tab2:
            st.plotly_chart(plot_freq(res.signals["tx"], aparams.fs, "Spectrum of modulated signal"), width='stretch')

        with tab3:
            st.json(res.meta)

        with tab4:
            st.write("Parameters:")
            st.json(res.meta)
