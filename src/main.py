"""
main.py — Unified CLI entry point for DeepPermNet-ViT

Usage:
    python main.py train     [--epochs N] [--batch-size N] [--lr FLOAT]
    python main.py evaluate  [--checkpoint PATH] [--num-batches N]
    python main.py visualize [--checkpoint PATH] [--num-samples N] [--output PATH]
"""

import sys
import os

# Make sure src/ is on the path so imports inside each module resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv.pop(1)   # consume the subcommand so argparse in each module sees a clean argv

    if command == "train":
        from train import main as run
    elif command == "evaluate":
        from evaluate import main as run
    elif command == "visualize":
        from visualize import main as run
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: train | evaluate | visualize")
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()