"""Keep executable scripts out of the top-level dependency search path."""
from pathlib import Path
import sys


def configure_cli_imports(root):
    """Support both ``python ops/x.py`` and ``python -m ops.x`` safely.

    Prepending the project root alone leaves the script's ops/ directory ahead
    of site-packages. For example, ops/kaggle.py then hides the real Kaggle SDK.
    Keep ops available as a package, never as a top-level dependency directory.
    Resolve empty/relative/symlink entries too, without removing other libraries.
    """
    root = Path(root).resolve()
    excluded = {root, root / "ops"}
    sys.path[:] = [str(root), *(entry for entry in sys.path
        if Path(entry or ".").resolve() not in excluded)]
