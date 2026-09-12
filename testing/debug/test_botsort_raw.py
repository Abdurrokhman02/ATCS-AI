import cv2
import argparse
import supervision as sv
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Debug raw YOLO -> BoT-SORT tracking"
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
        help="Confidence threshold YOLO"
    )

    parser.add_argument(
        "--output",
        default="testing/debug/botsort_id_diag.mp4",
        help="Output video"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Jumlah frame maksimum. Gunakan 0 untuk semua frame."
    )

    args = parser.parse_args()

    # ============================================================
    # HEADER
    # ============================================================

    print("=" * 70)
    print("RAW YOLO -> BOt-SORT DIAGNOSTIC")
    print("=" * 70)

    print(f"Model       : {args.model}")
    print(f"Source      : {args.source}")
    print(f"Confidence  : {args.conf}")
    print(f"Max frames  : {args.max_frames}")
    print()

    # ============================================================
    # LOAD YOLO
    # ============================================================

    print("Loading YOLO model...")

    model = YOLO(args.model)

    print("YOLO model loaded.")
    print()

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

    if fps <= 0:
        fps = 25.0

    print(f"Video FPS   : {fps}")
    print(f"Resolution  : {width}x{height}")
    print(f"Total frame : {total_frames}")
    print()

    # ============================================================
    # BoT-SORT
    #
    # NOTE:
    # Supervision 0.30.0 does not provide the same simple
    # sv.BoTSORT(...) interface as ByteTrack.
    #
    # Therefore this diagnostic uses Ultralytics' native
    # BoT-SORT tracker through model.track().
    #
    # ReID is NOT enabled.
    # ============================================================

    tracker_config = "botsort.yaml"

    print("BoT-SORT:")
    print(f"  tracker config : {tracker_config}")
    print(f"  ReID           : disabled")
    print(f"  video FPS      : {fps}")
    print()

    # ============================================================
    # OUTPUT VIDEO
    # ============================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Gagal membuat output video: {args.output}"
        )

    # ============================================================
    # STATE
    # ============================================================

    previous_ids = set()

    frame_idx = 0

    # ============================================================
    # TRACKING LOOP
    # ============================================================

    while True:

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()

        if not ret:
            break

        # ========================================================
        # YOLO + BoT-SORT
        #
        # persist=True:
        # mempertahankan state tracker antar frame.
        # ========================================================

        results = model.track(
            source=frame,
            conf=args.conf,
            persist=True,
            tracker=tracker_config,
            verbose=False,
        )

        result = results[0]

        # ========================================================
        # RAW YOLO DETECTIONS
        # ========================================================

        if result.boxes is not None:
            yolo_count = len(result.boxes)
        else:
            yolo_count = 0

        # ========================================================
        # TRACKED DETECTIONS
        # ========================================================

        tracked_boxes = result.boxes

        if (
            tracked_boxes is not None
            and tracked_boxes.id is not None
        ):
            tracker_ids = (
                tracked_boxes.id
                .int()
                .cpu()
                .tolist()
            )
        else:
            tracker_ids = []

        current_ids = {
            int(track_id)
            for track_id in tracker_ids
        }

        # ========================================================
        # CURRENT TRACK COUNT
        # ========================================================

        bot_count = len(current_ids)

        # ========================================================
        # DETECT ID CHANGES
        # ========================================================

        new_ids = current_ids - previous_ids
        disappeared = previous_ids - current_ids

        if disappeared or new_ids:

            print(
                f"[CHANGE frame={frame_idx:04d}] "
                f"YOLO={yolo_count:02d} "
                f"BOT={bot_count:02d} "
                f"DISAPPEARED={sorted(disappeared)} "
                f"NEW={sorted(new_ids)}"
            )

        # ========================================================
        # PERIODIC SUMMARY
        # ========================================================

        if frame_idx % 30 == 0:

            classes = []

            if (
                tracked_boxes is not None
                and tracked_boxes.cls is not None
            ):
                classes = [
                    int(class_id)
                    for class_id in
                    tracked_boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                ]

            print(
                f"[SUMMARY frame={frame_idx:04d}] "
                f"YOLO={yolo_count:02d} "
                f"BOT={bot_count:02d} "
                f"IDS={sorted(current_ids)} "
                f"CLASSES={classes}"
            )

        # ========================================================
        # DRAW RESULTS
        # ========================================================

        annotated = frame.copy()

        if (
            tracked_boxes is not None
            and len(tracked_boxes) > 0
        ):

            xyxy = (
                tracked_boxes.xyxy
                .cpu()
                .numpy()
            )

            confs = (
                tracked_boxes.conf
                .cpu()
                .numpy()
                if tracked_boxes.conf is not None
                else None
            )

            classes_tensor = (
                tracked_boxes.cls
                .int()
                .cpu()
                .numpy()
                if tracked_boxes.cls is not None
                else None
            )

            ids_tensor = (
                tracked_boxes.id
                .int()
                .cpu()
                .numpy()
                if tracked_boxes.id is not None
                else None
            )

            # ----------------------------------------------------
            # DRAW EACH TRACK
            # ----------------------------------------------------

            for i, box in enumerate(xyxy):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                if ids_tensor is not None:
                    track_id = int(ids_tensor[i])
                else:
                    track_id = -1

                # ------------------------------------------------
                # CLASS
                # ------------------------------------------------

                if classes_tensor is not None:
                    class_id = int(classes_tensor[i])
                else:
                    class_id = -1

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                if confs is not None:
                    confidence = float(confs[i])
                else:
                    confidence = 0.0

                # ------------------------------------------------
                # BOX
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

                (
                    text_w,
                    text_h
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )

                text_y = max(
                    y1 - 8,
                    text_h + baseline + 2
                )

                # ------------------------------------------------
                # LABEL BACKGROUND
                # ------------------------------------------------

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

                # ------------------------------------------------
                # LABEL TEXT
                # ------------------------------------------------

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
        # FRAME INFORMATION
        # ========================================================

        info = (
            f"Frame: {frame_idx} | "
            f"YOLO: {yolo_count} | "
            f"BoT-SORT: {bot_count}"
        )

        cv2.rectangle(
            annotated,
            (10, 10),
            (650, 45),
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
        # UPDATE TRACK STATE
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