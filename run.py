#!/usr/bin/env python3
"""HEDA-OD entry point (top-level runner).

Usage:
  python run.py --root_path ./dataset/ --data_path nyc24q1.npy ...
  # or: python -m models.HEDA_OD.run ...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.HEDA_OD.train import run

if __name__ == "__main__":
    run()
