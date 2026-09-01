import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = ("DETEKSI", "COUNTING", "KECEPATAN", "KEMACETAN", "LAWAN_ARAH", "INSIDEN", "ANPR")


def parse_args(default_source=None, require_source=False):
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=default_source)
    p.add_argument("--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "300")))
    p.add_argument("--model", default=os.getenv("MODEL_PATH", "./best_kendaraan.pt"))
    p.add_argument("--conf", default=os.getenv("MODEL_CONF", "0.25"))
    p.add_argument("--output-dir", default=os.getenv("TEST_OUTPUT_DIR", "./output/testing"))
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if require_source and not args.source:
        p.error("isi --source atau environment RTSP_URL")
    return args


def run(feature, source, max_frames=300, model="./best_kendaraan.pt", conf="0.25", output_dir="./output/testing", show=False):
    os.chdir(ROOT)
    import config
    for name in FEATURES:
        config.FLAGS[name] = name == feature
    if feature in ("LAWAN_ARAH", "INSIDEN"):
        config.FLAGS["DETEKSI"] = True
    if feature == "ANPR":
        config.FLAGS["DETEKSI"] = True
        config.FLAGS["LAWAN_ARAH"] = True
        config.FLAGS["INSIDEN"] = True
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_feature = feature.lower()
    argv = [
        "main.py",
        "--source", source,
        "--model", model,
        "--conf", str(conf),
        "--max-frames", str(max_frames),
        "--output-report", str(Path(output_dir) / f"{safe_feature}_report.json"),
        "--output-video", str(Path(output_dir) / f"{safe_feature}_output.mp4"),
    ]
    if not show:
        argv.append("--headless")
    sys.argv = argv
    import main
    main.main()
