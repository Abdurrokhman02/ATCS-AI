import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import parse_args, run


def main():
    args = parse_args(default_source="./samples/sampleuji2.mp4")
    run("KECEPATAN", args.source, args.max_frames, args.model, args.conf, args.output_dir, args.show)


if __name__ == "__main__":
    main()
