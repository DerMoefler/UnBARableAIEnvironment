# run_all.py: Helper to run all scripts in this scripts directory as well as its subdirectories (excluding the specified subdirectories)

import argparse
import pathlib
import subprocess
import sys


EXCLUDED_DIR_NAMES = {
    ".venv",
    "venv",
    "__pycache__",
    "generated",
    ".git",
}


def is_excluded(path: pathlib.Path, root: pathlib.Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in EXCLUDED_DIR_NAMES for part in relative_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all documentation image generators")

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where generated images should be stored",
    )

    args = parser.parse_args()

    scripts_dir = pathlib.Path(__file__).resolve().parent
    out_dir = pathlib.Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using output directory: {out_dir}\n")

    scripts = sorted(scripts_dir.rglob("*.py"))

    for script in scripts:
        if script.resolve() == pathlib.Path(__file__).resolve():
            continue

        if is_excluded(script, scripts_dir):
            continue

        print(f"[INFO] Running {script} via uv")

        result = subprocess.run(
            [
                "uv",
                "run",
                str(script),
                "--out-dir",
                str(out_dir),
            ],
            cwd=scripts_dir,
        )

        if result.returncode != 0:
            print(f"[ERROR] Script failed: {script}")
            sys.exit(result.returncode)

    print("\n[OK] All scripts completed successfully")


if __name__ == "__main__":
    main()