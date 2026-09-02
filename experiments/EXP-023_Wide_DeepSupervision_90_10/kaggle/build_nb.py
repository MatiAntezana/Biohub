import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
from aienv import ensure_aienv
ensure_aienv()

with open("inference.py", "r") as f:
    py_code = f.read()

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [py_code]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("submission_notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Generated submission_notebook.ipynb successfully!")
