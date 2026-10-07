# Python Attention Calculations with Rust

I decided to challenge myself and learn about how to integrate Rust and Python, two langauges I use interchangably. Thought linking them together would be a great learning oppertuntiy as both languages have unique advantages and disadvantages.

Additionally to see a comparison between Rust and Python, check out: `PyRust_Comparison.py`

## Installation guide

The included wheel contains the compiled Rust library, so you do not need to install Rust or Maturin.

The current wheel requires:

- CPython 3.12.
- An Apple Silicon Mac.
- macOS 11 or newer.

This wheel will not work on Windows, Linux, or an Intel Mac.

### Installing without Conda

Check that Python 3.12 is available:

```bash
python3.12 --version
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install NumPy and the included Rust library:

```bash
python -m pip install numpy
python -m pip install ./wheels/rustml-0.1.0-cp312-cp312-macosx_11_0_armg4.whl
```

Run the assignment:

```bash
python attention.py
```
