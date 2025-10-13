import argparse
from pathlib import Path

from checkpoint_utils import load_msgpack_checkpoint, save_model_state_to_npz


def convert_checkpoint(source: Path, destination: Path, overwrite: bool = False) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint {source} does not exist.")

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination {destination} already exists. Use --overwrite to replace it.")

    checkpoint_data = load_msgpack_checkpoint(str(source))
    if "model_state" not in checkpoint_data:
        raise KeyError(f"Checkpoint {source} does not contain a model_state.")

    model_state = checkpoint_data["model_state"]
    step = checkpoint_data.get("step", 0)

    save_model_state_to_npz(model_state, step, str(destination))
    print(f"Converted {source} -> {destination} (step {step})")


def main():
    parser = argparse.ArgumentParser(description="Convert msgpack checkpoints to NPZ format.")
    parser.add_argument("source", type=Path, help="Path to the .msgpack checkpoint file.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Output path for the NPZ checkpoint. Defaults to replacing the extension with .npz.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")

    args = parser.parse_args()
    source_path = args.source
    destination_path = args.destination if args.destination else source_path.with_suffix(".npz")

    convert_checkpoint(source_path, destination_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
