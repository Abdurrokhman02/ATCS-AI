# main.py
import os
import warnings
import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from config import (
    CAM_CONFIG, GARIS, KEMACETAN_CONFIG, LAWAN_ARAH_CONFIG,
    INSIDEN_CONFIG, ANPR_CONFIG, OUTPUT_CONFIG,
    MODEL_CONFIG, TRACKER_CONFIG, SMOOTHER_CONFIG,
    FLAGS, BUTTONS, CLASS_NAMES,
)
from modules.fitur_counting import VehicleCounter
from modules.fitur_kecepatan import hitung_kecepatan_kendaraan, get_rata_rata_kecepatan
from modules.fitur_kemacetan import cek_status_kemacetan
from modules.fitur_lawan_arah import WrongWayDetector
from modules.fitur_insiden import IncidentDetector
from modules.event_logger import EventLogger
from modules.fitur_stabilisasi import BoxPersistence

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="ATCS Dishub KBB - AI Traffic Monitoring")
    p.add_argument("--source", default=CAM_CONFIG["video_source"],
                   help="Path video atau URL RTSP kamera")
    p.add_argument("--model", default=MODEL_CONFIG["path"],
                   help="Path model YOLO")
    p.add_argument("--conf", type=float, default=MODEL_CONFIG["conf"],
                   help="Confidence threshold deteksi")
    p.add_argument("--headless", action="store_true",
                   help="Tanpa GUI (untuk testing / CI)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Batasi jumlah frame yang diproses (0 = semua)")
    p.add_argument("--output-video", default="",
                   help="Path output video terannotasi (.mp4)")
    p.add_argument("--output-report", default="",
                   help="Path output laporan statistik JSON")
    p.add_argument("--no-lawan-arah", action="store_true", help="Nonaktifkan modul Lawan Arah")
    p.add_argument("--no-insiden", action="store_true", help="Nonaktifkan modul Insiden")
    return p.parse_args()


def handle_mouse_click(event, x, y, flags_param, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for feature_name, (x1, y1, x2, y2) in BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                FLAGS[feature_name] = not FLAGS[feature_name]


def find_box_by_tid(detections, tid):
    if detections.tracker_id is None:
        return None
    idx = np.where(detections.tracker_id == tid)[0]
    if len(idx) == 0:
        return None
    return detections.xyxy[idx[0]]


def main():
    args = parse_args()
    headless = args.headless

    if args.no_lawan_arah:
        FLAGS["LAWAN_ARAH"] = False
    if args.no_insiden:
        FLAGS["INSIDEN"] = False

    FLAGS["ANPR"] = False
    ANPR_CONFIG["enabled"] = False

    # ---------- KAMERA ----------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video: {args.source}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ---------- GARIS VIRTUAL ----------
    garis1_y = int(height * GARIS["garis1_y_frac"])
    garis2_y = int(height * GARIS["garis2_y_frac"])
    LINE1 = sv.LineZone(start=sv.Point(0, garis1_y), end=sv.Point(width, garis1_y))
    LINE2 = sv.LineZone(start=sv.Point(width, garis2_y), end=sv.Point(0, garis2_y))

    MAX_ROI = np.array([[0, garis1_y], [width, garis1_y], [width, garis2_y], [0, garis2_y]])
    try:
        roi_zone = sv.PolygonZone(polygon=MAX_ROI, frame_resolution_wh=(width, height))
    except TypeError:
        roi_zone = sv.PolygonZone(polygon=MAX_ROI)

    # ---------- AI & TRACKER ----------
    model = YOLO(args.model)
    smoother = sv.DetectionsSmoother(length=SMOOTHER_CONFIG["length"])
    box_persist = BoxPersistence(
        grace_frames=SMOOTHER_CONFIG["persist_grace_frames"],
        decay=SMOOTHER_CONFIG["persist_decay"],
        min_detections=SMOOTHER_CONFIG["persist_min_detections"],
    )

    # For diagnostic tracking of IDs across frames
    prev_tracker_ids = set()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    l1_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.GREEN)
    l2_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.RED)

    # ---------- STATE MODUL ----------
    counter = VehicleCounter(arah_lalu_lintas=CAM_CONFIG["arah_lalu_lintas"])
    wrongway = WrongWayDetector(
        arah_lalu_lintas=CAM_CONFIG["arah_lalu_lintas"],
        min_kecepatan_px=LAWAN_ARAH_CONFIG["min_kecepatan_px"],
        frame_konfirmasi=LAWAN_ARAH_CONFIG["frame_konfirmasi"],
        cooldown_detik=LAWAN_ARAH_CONFIG["cooldown_detik"],
    )
    incident = IncidentDetector(INSIDEN_CONFIG)
    anpr = None
    logger = EventLogger(OUTPUT_CONFIG["event_dir"],
                         simpan_snapshot=OUTPUT_CONFIG["simpan_snapshot"],
                         camera_id=CAM_CONFIG["camera_id"])

    speeds_kmh = []
    recent_events = deque(maxlen=4)
    total_event_count = 0
    frame_idx = 0
    t_mulai = cv2.getTickCount()

    # ---------- VIDEO OUTPUT ----------
    out_writer = None
    if args.output_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        out_writer = cv2.VideoWriter(args.output_video,
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Statistik laporan
    speed_total = 0.0
    speed_count = 0
    kongesti_distribusi = {}   # status -> jumlah frame

    # ---------- WINDOW (GUI mode) ----------
    if not headless:
        cv2.namedWindow("CCTV Traffic Monitor")
        cv2.setMouseCallback("CCTV Traffic Monitor", handle_mouse_click)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break
            
        # ---> PERUBAHAN DI SINI <---
        # Jika argumen source adalah file video (mengandung .mp4/.avi/dll), pakai simulasi waktu frame.
        # Jika berupa RTSP atau angka webcam (0, 1), pakai waktu nyata (time.time()).
        if any(ext in args.source.lower() for ext in [".mp4", ".avi", ".mkv", ".mov"]):
            t_sekarang = frame_idx / fps
        else:
            import time
            # Gunakan waktu relatif sejak program jalan atau timestamp aktual
            if 'waktu_mulai_realtime' not in locals():
                waktu_mulai_realtime = time.time()
            t_sekarang = time.time() - waktu_mulai_realtime
            
        # 1. AI Inference (deteksi + tracking) - Native Ultralytics BoT-SORT
        # Single inference call with integrated tracking (no separate YOLO + ByteTrack)
        results = model.track(
            source=frame,
            conf=args.conf,
            persist=True,
            tracker="botsort.yaml",
            classes=[1, 2, 3, 4],
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Ensure tracker_id is populated from BoT-SORT (boxes.id)
        if detections.tracker_id is None and hasattr(results.boxes, 'id') and results.boxes.id is not None:
            detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)

        n_botsort = len(detections)

        # Diagnostics: track ID changes
        curr_tracker_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        new_ids = curr_tracker_ids - prev_tracker_ids
        lost_ids = prev_tracker_ids - curr_tracker_ids
        prev_tracker_ids = curr_tracker_ids

        # Persistence output (for counting & line crossing)
        detections_persist = box_persist.update(detections, frame_idx)
        n_persist_out = len(detections_persist)

        # Smoother output (for rendering & other analytics)
        if len(detections_persist):
            detections_smooth = smoother.update_with_detections(detections_persist)
        else:
            detections_smooth = detections_persist
        n_smooth_out = len(detections_smooth)

        # For backward compatibility with existing diagnostic variable names
        n_byte_track = n_botsort
        ghost_count = max(0, n_persist_out - n_botsort)

        # For diagnostic output - use accurate labels
        n_tracked = n_botsort  # BoT-SORT tracked detections

        # 2. Trigger Garis Virtual - USE PERSISTENCE OUTPUT (non-smoothed) for accurate crossing detection
        c1_in, c1_out = LINE1.trigger(detections=detections_persist)
        c2_in, c2_out = LINE2.trigger(detections=detections_persist)
        crossed_l1 = c1_in | c1_out
        crossed_l2 = c2_in | c2_out

        # 3. Logika per kendaraan - COUNTING uses persistence output
        if detections_persist.tracker_id is not None:
            for i, (tid, cid) in enumerate(zip(detections_persist.tracker_id, detections_persist.class_id)):
                hitung_kecepatan_kendaraan(
                    tid, crossed_l1[i], crossed_l2[i], t_sekarang,
                    counter.timestamp_l1, counter.timestamp_l2,
                    speeds_kmh, CAM_CONFIG["jarak_garis_meter"],
                )
                counter.update(tid, cid, c1_in[i], c1_out[i], c2_in[i], c2_out[i], t_sekarang)

        # 4. Modul AI lanjutan (reuse SMOOTHED detections for rendering & other analytics)
        wrong_mask, ww_events = wrongway.update(detections_smooth, t_sekarang)
        incident_events = incident.update(detections_persist, t_sekarang, frame=frame, frame_idx=frame_idx)

        total_akumulasi = counter.get_total_akumulasi()
        in_count, out_count = counter.get_in_out()
        kendaraan_live = len(detections_smooth)
        avg_speed = get_rata_rata_kecepatan(speeds_kmh)

        in_roi_mask = roi_zone.trigger(detections=detections_smooth)
        jumlah_di_roi = int(in_roi_mask.sum()) if len(in_roi_mask) else 0
        status_macet, color_macet = cek_status_kemacetan(
            jumlah_di_roi, CAM_CONFIG["panjang_segmen_meter"],
            CAM_CONFIG["jumlah_lajur"], avg_speed,
            CAM_CONFIG["free_flow_speed_kmh"], KEMACETAN_CONFIG,
        )
        kongesti_distribusi[status_macet] = kongesti_distribusi.get(status_macet, 0) + 1
        if avg_speed > 0.0:
            speed_total += avg_speed
            speed_count += 1

        # 5. Event logging + ANPR trigger
        if FLAGS["LAWAN_ARAH"]:
            for ev in ww_events:
                box = find_box_by_tid(detections_smooth, ev["tid"])
                snap = _crop_by_box(frame, box)
                metadata = {"tracker_id": ev["tid"]}
                total_event_count += 1
                logger.log_event("LAWAN_ARAH", ev["confidence"], snap, metadata=metadata)
                recent_events.append(f"LAWAN ARAH #{ev['tid']}")

        if FLAGS["INSIDEN"]:
            for ev in incident_events:
                box = find_box_by_tid(detections_smooth, ev["tid"])
                snap = _crop_by_box(frame, box)
                metadata = {"tracker_id": ev["tid"]}
                total_event_count += 1
                logger.log_event(ev["tipe"], ev["confidence"], snap, metadata=metadata)
                recent_events.append(f"{ev['tipe']} #{ev['tid']}")

        # 6. RENDER - use smoothed detections for visual stability
        annotated = frame.copy()    

        if FLAGS["DETEKSI"]:
            if len(detections_smooth):
                annotated = box_annotator.annotate(scene=annotated, detections=detections_smooth)
                labels = [f"#{tid} {CLASS_NAMES.get(cid, '?')}"
                          for tid, cid in zip(detections_smooth.tracker_id, detections_smooth.class_id)]
                annotated = label_annotator.annotate(scene=annotated, detections=detections_smooth, labels=labels)

        if FLAGS["LAWAN_ARAH"]:
            annotated = _render_wrong_way(annotated, detections_smooth, wrong_mask)

        if FLAGS["COUNTING"]:
            try:
                annotated = l1_annotator.annotate(annotated, line_zone=LINE1)
                annotated = l2_annotator.annotate(annotated, line_zone=LINE2)
            except TypeError:
                annotated = l1_annotator.annotate(annotated, line_counter=LINE1)
                annotated = l2_annotator.annotate(annotated, line_counter=LINE2)

        _render_ui(annotated, total_akumulasi, in_count, out_count, kendaraan_live,
                   avg_speed, status_macet, color_macet, recent_events)

        if out_writer is not None:
            out_writer.write(annotated)

        # 7. Housekeeping
        # 7. Housekeeping
        if frame_idx % (int(fps) * 5) == 0:
            counter.bersihkan_memori(t_sekarang)
            wrongway.bersihkan_memori(t_sekarang)
            incident.bersihkan_memori(t_sekarang)
            
            if len(speeds_kmh) > 200:
                del speeds_kmh[:-100]

        if headless:
            if frame_idx % max(1, int(fps * 2)) == 0:
                ids_str = ",".join(map(str, sorted(curr_tracker_ids))) if curr_tracker_ids else "-"
                new_str = ",".join(map(str, sorted(new_ids))) if new_ids else "-"
                lost_str = ",".join(map(str, sorted(lost_ids))) if lost_ids else "-"
                print(f"[{frame_idx:6d}] t={t_sekarang:5.1f}s "
                      f"tracked={n_tracked:2d} persist={n_persist_out:2d} smooth={n_smooth_out:2d} "
                      f"ids=[{ids_str}] new=[{new_str}] lost=[{lost_str}] "
                      f"live={kendaraan_live:2d} total={total_akumulasi:4d} "
                      f"avg={avg_speed:5.1f} km/h {status_macet:6s} events={total_event_count}")
            if frame_idx % max(1, int(fps * 10)) == 0:
                c1 = int(crossed_l1.sum()) if len(crossed_l1) else 0
                c2 = int(crossed_l2.sum()) if len(crossed_l2) else 0
                print(f"  DIAG: cross_L1={c1} cross_L2={c2} "
                      f"speeds_sample={len(speeds_kmh)} roi_count={jumlah_di_roi}")
        else:
            cv2.imshow("CCTV Traffic Monitor", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                FLAGS["DETEKSI"] = not FLAGS["DETEKSI"]
            elif key == ord('2'):
                FLAGS["COUNTING"] = not FLAGS["COUNTING"]
            elif key == ord('3'):
                FLAGS["KECEPATAN"] = not FLAGS["KECEPATAN"]
            elif key == ord('4'):
                FLAGS["KEMACETAN"] = not FLAGS["KEMACETAN"]
            elif key == ord('5'):
                FLAGS["LAWAN_ARAH"] = not FLAGS["LAWAN_ARAH"]
            elif key == ord('6'):
                FLAGS["INSIDEN"] = not FLAGS["INSIDEN"]

        frame_idx += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if not headless:
        cv2.destroyAllWindows()

    dt = (cv2.getTickCount() - t_mulai) / cv2.getTickFrequency()
    rata_kecepatan_keseluruhan = (speed_total / speed_count) if speed_count else 0.0
    print("\n================ RINGKASAN ================")
    print(f"Kamera            : {CAM_CONFIG['camera_id']}")
    print(f"Frame diproses    : {frame_idx} ({(frame_idx / dt):.1f} fps)" if dt > 0 else f"Frame diproses : {frame_idx}")
    print(f"Total Kendaraan   : {total_akumulasi}")
    print(f"  - Masuk (in)    : {in_count}")
    print(f"  - Keluar (out)  : {out_count}")
    print(f"  Per jenis       : {dict(counter.get_per_kelas())}")
    print(f"Rata2 Kecepatan   : {avg_speed:.1f} km/jam (rata-rata periode berjalan)")
    print(f"  - Rerata seluruh sampel : {rata_kecepatan_keseluruhan:.1f} km/jam")
    print(f"Status Kemacetan  : {status_macet} (density {jumlah_di_roi} kend. di ROI)")
    print(f"  Distribusi      : {kongesti_distribusi}")
    print(f"Event dicatat     : {total_event_count} (lihat {OUTPUT_CONFIG['event_dir']})")
    if args.output_video:
        print(f"Video output      : {args.output_video}")
    print("===========================================")

    if args.output_report:
        report = {
            "camera_id": CAM_CONFIG["camera_id"],
            "source": args.source,
            "model": args.model,
            "frame_count": frame_idx,
            "processing_fps": round(frame_idx / dt, 2) if dt > 0 else 0.0,
            "total_kendaraan": total_akumulasi,
            "in": in_count,
            "out": out_count,
            "per_jenis": {str(k): int(v) for k, v in counter.get_per_kelas().items()},
            "rata_rata_kecepatan_kmh": round(avg_speed, 2),
            "rata_rata_kecepatan_keseluruhan_kmh": round(rata_kecepatan_keseluruhan, 2),
            "status_kemacetan_akhir": status_macet,
            "distribusi_kemacetan": kongesti_distribusi,
            "event_count": total_event_count,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        import json
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Laporan disimpan  : {args.output_report}")


def _crop_by_box(frame, box):
    if box is None:
        return frame
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _proses_anpr(anpr, tid, vehicle_crop, t_sekarang, trigger, recent_events):
    return {"plat": "", "backend": None, "anpr_conf": 0.0}


def _render_wrong_way(frame, detections, wrong_mask):
    """Gambar bounding box merah + label peringatan untuk kendaraan melawan arah."""
    if detections.tracker_id is None or not wrong_mask.any():
        return frame
    for i, (tid, xyxy) in enumerate(zip(detections.tracker_id, detections.xyxy)):
        if not wrong_mask[i]:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"LAWAN ARAH #{int(tid)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    return frame


def _render_ui(frame, total, in_count, out_count, live, avg_speed, status, color_macet, events):
    """Render tombol menu, panel dashboard, dan strip event terbaru."""
    for btn_name, (x1, y1, x2, y2) in BUTTONS.items():
        is_active = FLAGS[btn_name]
        color = (0, 180, 0) if is_active else (60, 60, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, f"{btn_name}: {'ON' if is_active else 'OFF'}",
                    (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 60), (440, 245), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    cv2.putText(frame, "DISHUB - TRAFFIC MONITORING", (35, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    v_akum = f"{total} unit" if FLAGS["COUNTING"] else "[NONAKTIF]"
    v_inout = f"In:{in_count} Out:{out_count}" if FLAGS["COUNTING"] else "[NONAKTIF]"
    v_live = f"{live} unit" if FLAGS["DETEKSI"] else "[NONAKTIF]"
    v_sped = f"{avg_speed:.1f} km/jam" if FLAGS["KECEPATAN"] else "[NONAKTIF]"
    v_mact = status if FLAGS["KEMACETAN"] else "[NONAKTIF]"
    c_mact = color_macet if FLAGS["KEMACETAN"] else (150, 150, 150)

    cv2.putText(frame, f"1. Total Akumulasi  : {v_akum}", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"2. Arah (In/Out)    : {v_inout}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
    cv2.putText(frame, f"3. Live Frame       : {v_live}", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"4. Rata2 Kecepatan  : {v_sped}", (35, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(frame, f"5. Status Kemacetan : {v_mact}", (35, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_mact, 2)

    if events:
        ev_text = " | ".join(events)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (20, h - 45), (620, h - 12), (0, 0, 0), -1)
        cv2.putText(frame, f"EVENT: {ev_text}", (28, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


if __name__ == "__main__":
    main()
