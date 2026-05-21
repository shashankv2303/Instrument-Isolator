import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


# All stems available from the htdemucs_6s model
VALID_STEMS = {"drums", "bass", "other", "vocals", "guitar", "piano"}


def separate(
    input_file: str,
    stems: list[str],
    output_dir: str = "output",
) -> dict[str, Path]:
    """Separate and optionally combine stems from a song.

    Returns a dict of stem name -> path. If multiple stems are requested,
    a "combined" key is also included with the mixed-down file.

    stems: any combination of "drums", "bass", "other", "vocals", "guitar", "piano"
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    invalid = set(stems) - VALID_STEMS
    if invalid:
        raise ValueError(f"Invalid stems: {invalid}. Choose from: {VALID_STEMS}")
    if not stems:
        raise ValueError("Provide at least one stem.")

    wrapper = Path(__file__).parent / "demucs_wrapper.py"
    subprocess.run(
        [
            sys.executable, str(wrapper),
            "-n", "htdemucs_6s",
            "--overlap", "0.1",
            "--out", output_dir,
            str(input_path),
        ],
        check=True,
    )

    stems_dir = Path(output_dir) / "htdemucs_6s" / input_path.stem
    out = Path(output_dir)
    paths: dict[str, Path] = {}

    for stem in stems:
        src = stems_dir / f"{stem}.wav"
        if not src.exists():
            raise FileNotFoundError(f"Stem not found: {src}")
        dest = out / f"{input_path.stem}_{stem}.wav"
        shutil.move(str(src), dest)
        paths[stem] = dest

    if len(paths) > 1:
        stem_name = "_".join(stems)
        combined_path = out / f"{input_path.stem}_{stem_name}.wav"
        mixed: np.ndarray | None = None
        sr: int = 0
        for path in paths.values():
            audio, sr = sf.read(str(path), dtype="float32")
            mixed = audio if mixed is None else mixed + audio
        sf.write(str(combined_path), mixed, sr)
        paths["combined"] = combined_path

    return paths


if __name__ == "__main__":
    audio_file = input("Enter path to audio file: ").strip().strip('"')
    if not Path(audio_file).exists():
        print(f"File not found: {audio_file}")
        sys.exit(1)
    out_dir = "output"

    STEM_LIST = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    print("\nAvailable stems:")
    for i, stem in enumerate(STEM_LIST, 1):
        print(f"  {i}. {stem}")

    print("\nEnter stem numbers to extract.")
    print("  Single:   5       → guitar only")
    print("  Combined: 1+2     → drums + bass mixed into one file")
    print("  Both:     1,2     → drums and bass as separate files")
    print("  Mix:      1+2,5   → drums+bass combined, guitar separate")

    raw = input("\nYour selection: ").strip()

    # Parse comma-separated groups; within each group "+" means combine
    requested_stems: list[str] = []
    groups: list[list[str]] = []

    for group in raw.split(","):
        indices = group.strip().split("+")
        group_stems = []
        for idx in indices:
            idx = idx.strip()
            if not idx.isdigit() or not (1 <= int(idx) <= len(STEM_LIST)):
                print(f"Invalid selection: '{idx}'. Must be 1–{len(STEM_LIST)}.")
                sys.exit(1)
            group_stems.append(STEM_LIST[int(idx) - 1])
        groups.append(group_stems)
        requested_stems.extend(group_stems)

    requested_stems = list(dict.fromkeys(requested_stems))  # deduplicate, preserve order

    print(f"\nExtracting: {', '.join(requested_stems)}\n")
    paths = separate(audio_file, requested_stems, out_dir)

    # If the user specified combined groups, produce those mixes
    for group in groups:
        if len(group) > 1:
            group_name = "_".join(group)
            combined_path = Path(out_dir) / f"{Path(audio_file).stem}_{group_name}.wav"
            if paths.get("combined") and len(groups) == 1:
                # already created by separate()
                pass
            else:
                mixed: np.ndarray | None = None
                sr: int = 0
                for stem in group:
                    audio, sr = sf.read(str(paths[stem]), dtype="float32")
                    mixed = audio if mixed is None else mixed + audio
                sf.write(str(combined_path), mixed, sr)
                paths[f"combined ({group_name})"] = combined_path

    print("\nOutput files:")
    for name, path in paths.items():
        if name != "combined" or len(groups) == 1:
            print(f"  {name}: {path}")
