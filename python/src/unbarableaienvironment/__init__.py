from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Standardpfad des gemeinsamen Projekt-Users
# Optional kann BAR_AI_SHARED_ROOT gesetzt werden, um den Pfad zu überschreiben.
_SHARED_ROOT = Path(
    os.environ.get(
        "BAR_AI_SHARED_ROOT",
        "/home/projekt/repos/UnBARableAIEnvironment.git",
    )
).resolve()

_py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
_platform_dir = f"Linux-x86_64"  # bei Bedarf später dynamisch machen

_native_dir = _SHARED_ROOT / "artifacts" / _platform_dir / _py_tag
_candidates = sorted(_native_dir.glob("bar_ai*.so"))

if not _candidates:
    raise ImportError(
        "Keine native bar_ai-Bibliothek gefunden.\n"
        f"Gesucht wurde in: {_native_dir}\n"
        "Erwartet wurde z. B. bar_ai.cpython-312-...so\n"
        "Bitte sicherstellen, dass der Projekt-User die C++-Bindings gebaut hat."
    )

_native_path = _candidates[0]

_spec = importlib.util.spec_from_file_location("bar_ai.bar_ai", _native_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Konnte Native-Modul nicht laden: {_native_path}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

Unit = _module.Unit
UnitData = _module.UnitData
Pawn = _module.Pawn

__all__ = ["Unit", "UnitData", "Pawn"]