import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Debug raw YOLO -> ByteTrack tracking"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path ke YOLO model"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path video source"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--output",
        default="testing/debug/bytetrack_id_diag.mp4",
        help="Output video"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Jumlah frame maksimum"
    )

    args = parser.parse_args()

    # ============================================================
    # LOAD MODEL
    # ============================================================

    print("=" * 70)
    print("RAW YOLO -> BYTETRACK DIAGNOSTIC")
    print("=" * 70)

    print(f"Model       : {args.model}")
    print(f"Source      : {args.source}")
    print(f"Confidence  : {args.conf}")
    print(f"Max frames  : {args.max_frames}")
    print()

    model = YOLO(args.model)

    # ============================================================
    # OPEN VIDEO
    # ============================================================

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Gagal membuka video: {args.source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS   : {fps}")
    print(f"Resolution  : {width}x{height}")
    print(f"Total frame : {total_frames}")
    print()

    # ============================================================
    # BYTE TRACK
    # ============================================================

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=45,
        minimum_matching_threshold=0.3,
        frame_rate=round(fps) if fps > 0 else 25,
    )

    print("ByteTrack:")
    print(f"  activation threshold : 0.25")
    print(f"  lost track buffer    : 45 frames")
    print(f"  matching threshold   : 0.3")
    print(f"  tracker FPS          : {round(fps) if fps > 0 else 25}")
    print()

    # ============================================================
    # OUTPUT VIDEO
    # ============================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps if fps > 0 else 25,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Gagal membuat output video: {args.output}"
        )

    # ============================================================
    # STATE
    # ============================================================

    previous_ids = set()

    frame_idx = 0

    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()

        if not ret:
            break

        # ========================================================
        # YOLO RAW
        # ========================================================

        results = model.predict(
            frame,
            conf=args.conf,
            verbose=False,
        )

        result = results[0]

        detections = sv.Detections.from_ultralytics(result)

        # ========================================================
        # IMPORTANT:
        # NO CLASS FILTER
        #
        # Kita sengaja tidak filter class dulu.
        # Tujuannya memastikan apakah masalah berasal dari
        # ByteTrack atau dari filter kendaraan.
        # ========================================================

        tracked = tracker.update_with_detections(
            detections
        )

        # ========================================================
        # CURRENT TRACK IDS
        # ========================================================

        current_ids = set()

        if tracked.tracker_id is not None:
            current_ids = {
                int(tid)
                for tid in tracked.tracker_id
            }

        # ========================================================
        # DETECT ID CHANGES
        # ========================================================

        new_ids = current_ids - previous_ids
        disappeared = previous_ids - current_ids

        if disappeared or new_ids:

            print(
                f"[CHANGE frame={frame_idx:04d}] "
                f"YOLO={len(detections):02d} "
                f"BYTE={len(tracked):02d} "
                f"DISAPPEARED={sorted(disappeared)} "
                f"NEW={sorted(new_ids)}"
            )

        # ========================================================
        # PERIODIC SUMMARY
        # ========================================================

        if frame_idx % 30 == 0:

            classes = []

            if detections.class_id is not None:
                classes = [
                    int(cid)
                    for cid in detections.class_id
                ]

            print(
                f"[SUMMARY frame={frame_idx:04d}] "
                f"YOLO={len(detections):02d} "
                f"BYTE={len(tracked):02d} "
                f"IDS={sorted(current_ids)} "
                f"CLASSES={classes}"
            )

        # ========================================================
        # DRAW TRACKING RESULT
        # ========================================================

        annotated = frame.copy()

        if len(tracked) > 0:

            boxes = tracked.xyxy

            class_ids = tracked.class_id

            confidences = tracked.confidence

            tracker_ids = tracked.tracker_id

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # ------------------------------------------------
                # CLASS
                # ------------------------------------------------

                if class_ids is not None:
                    class_id = int(class_ids[i])
                else:
                    class_id = -1

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                if confidences is not None:
                    confidence = float(confidences[i])
                else:
                    confidence = 0.0

                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                if tracker_ids is not None:
                    track_id = int(tracker_ids[i])
                else:
                    track_id = -1

                # ------------------------------------------------
                # DRAW BOX
                # ------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"ID:{track_id} "
                    f"class:{class_id} "
                    f"{confidence:.2f}"
                )

                # ------------------------------------------------
                # TEXT BACKGROUND
                # ------------------------------------------------

                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )

                text_y = max(
                    y1 - 8,
                    text_h + baseline + 2
                )

                cv2.rectangle(
                    annotated,
                    (
                        x1,
                        text_y - text_h - baseline - 2
                    ),
                    (
                        x1 + text_w + 4,
                        text_y
                    ),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1 + 2,
                        text_y - baseline - 1
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # ========================================================
        # FRAME INFO
        # ========================================================

        info = (
            f"Frame: {frame_idx} | "
            f"YOLO: {len(detections)} | "
            f"ByteTrack: {len(tracked)}"
        )

        cv2.rectangle(
            annotated,
            (10, 10),
            (620, 45),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            info,
            (15, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ========================================================
        # WRITE OUTPUT
        # ========================================================

        writer.write(annotated)

        # ========================================================
        # UPDATE STATE
        # ========================================================

        previous_ids = current_ids.copy()

        frame_idx += 1

    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print("DIAGNOSTIC SELESAI")
    print("=" * 70)
    print(f"Frames processed : {frame_idx}")
    print(f"Output           : {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()