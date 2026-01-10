# Encoding and Modulation Techniques Simulator

Interactive Streamlit-based simulator for common techniques in computer communications:

- Digital-to-Digital line coding (NRZ-L, NRZI, Manchester, Differential Manchester, AMI, Pseudoternary)
- Scrambling (B8ZS, HDB3)
- Digital-to-Analog modulation (ASK, FSK, PSK, QPSK, 16-QAM) _(in progress / if included)_

## Run locally

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the app

```bash
streamlit run comm_sim/app.py
```

### To run the Tests

```bash
pytest -q
```

### Notes

This project was developed for BLG 337E (Principles of Computer Communications), and is intended as an educational simulator.
