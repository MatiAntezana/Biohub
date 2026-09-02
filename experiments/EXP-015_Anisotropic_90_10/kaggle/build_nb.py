import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
from aienv import ensure_aienv
ensure_aienv()

KAGGLE_DIR = Path(__file__).resolve().parent
py_code = (KAGGLE_DIR / "inference.py").read_text(encoding="utf-8")
notebook = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [py_code]}],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
(KAGGLE_DIR / "submission_notebook.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Generated submission_notebook.ipynb successfully!")
