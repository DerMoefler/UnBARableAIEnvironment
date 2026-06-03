# template.py: A template for visualization scripts.

import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for generated image"
    )

    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)

    # Your output file
    output_file = out_dir / "my_diagram.png"

    # Example: generate something
    print(f"Writing to {output_file}")

    # Replace with real generation logic
    with open(output_file, "wb") as f:
        f.write(b"")  # placeholder

if __name__ == "__main__":
    main()
