# matcher (research skeleton)
A lightweight research-ready skeleton for feature matching and evaluation.

## Quick start
1. Create a venv and install requirements.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install package:
   ```bash
   pip install -e src
   ```
3. Run demo:
   ```bash
   python scripts/run_demo.py
   ```
4. Output visualization will be in `results/demo_matches.png`.

## Structure
- `src/mymatch` core modules (detector, matcher, filter, API)
- `scripts` quick scripts to run demo / evaluation
- `examples` sample images
- `results` output images
