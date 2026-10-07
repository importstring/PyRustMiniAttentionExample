# Python Attention Calculations with Rust

https://github.com/importstring/PyRustMiniAttentionExample \n
I decided to challenge myself and learn about how to integrate Rust and Python, two langauges I use interchangably. Thought linking them together would be a great learning oppertuntiy as both languages have unique advantages and disadvantages.

Additionally to see a comparison between Rust and Python, check out: `PyRust_Comparison.py`

## Installation guide

This assignment includes a precompiled version of my `rustml`
library. Rust, Cargo, and Maturin are not required to run it.

### Requirements

- A Mac with an Intel or Apple Silicon processor.
- CPython 3.12.
- An internet connection to install NumPy.
- A Python installation with working Turtle/Tk support for the
  optional graphical visualization.

The included Rust wheel contains builds for both Intel and
Apple Silicon Macs. Windows and Linux are not supported by
this submission.

The wheel's platform tags specify macOS 10.12 or newer for Intel
and macOS 11 or newer for Apple Silicon. Python, NumPy, and their
dependencies may require a newer macOS version.

Extract the ZIP and open a terminal inside the extracted project
folder before following either installation method.

### Installing without Conda

Check that Python 3.12 is installed:

```bash
python3.12 --version
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install NumPy and the included wheel:

```bash
python -m pip install numpy ./wheels/rustml-0.1.0-cp312-cp312-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl
```

### Installing with Conda

If Conda is already installed, create and activate an environment:

```bash
conda create -n AttentionDemo python=3.12 pip
conda activate AttentionDemo
```

Install NumPy and the included wheel:

```bash
python -m pip install numpy ./wheels/rustml-0.1.0-cp312-cp312-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl
```

### Running the programs

Run the attention calculation example:

```bash
python attention.py
```

Enter `y` when prompted to open the visualization, or `n` to exit.

Visualization controls:

- Space or Right arrow: next step
- Left arrow: previous step
- Home: restart
- Escape: close

Run the Python-list versus Rust comparison:

```bash
python PyRust_Comparison.py
```

This compares transposition, matrix multiplication, scaled scores,
and row-wise softmax outputs. It checks numerical agreement, not speed.
Scaling uses the same Python implementation in both paths.

If using VS Code, select the same Python environment used for
installation before running either script.

### Testing and limitations

The earlier Apple Silicon wheel was installed and tested in a fresh
Python environment. Both programs ran, and all four numerical
comparisons returned `True`.

The replacement universal2 wheel was built successfully for Intel
and Apple Silicon. It has not been tested on an Intel Mac.

The example demonstrates attention-score calculations and softmax
weights. It is not a complete attention layer or a trained model.
