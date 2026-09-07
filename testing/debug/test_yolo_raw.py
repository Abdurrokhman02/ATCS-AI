import cv2
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", default="testing/debug/yolo_raw.mp4")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO RAW DEBUG")
    print("=" * 60)
    print(f"Model : {args.model}")
    print(f"Source: {args.source}")
    print(f"Conf  : {args.conf}")
    print()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video : {width}x{height}")
    print(f"FPS   : {fps}")
    print()

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (width, height),
    )

    frame_idx = 0

    while frame_idx < args.max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=args.conf,
            verbose=False,
        )[0]

        boxes = results.boxes

        total = 0

        if boxes is not None:
            total = len(boxes)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"class={cls} conf={conf:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.putText(
            frame,
            f"YOLO RAW | detections={total} | conf={args.conf}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        writer.write(frame)

        if frame_idx % 30 == 0:
            classes = []

            if boxes is not None:
                classes = [int(x) for x in boxes.cls.tolist()]

            print(
                f"[frame {frame_idx:04d}] "
                f"detections={total} "
                f"classes={classes}"
            )

        frame_idx += 1

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("SELESAI")
    print(f"Output: {args.output}")
    print(f"Frames: {frame_idx}")
    print("=" * 60)


if __name__ == "__main__":
    main()