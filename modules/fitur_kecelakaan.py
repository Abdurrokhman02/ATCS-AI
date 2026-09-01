# modules/fitur_kecelakaan.py
import os
import cv2

# Konstanta dari file referensi
BUFFER_SIZE = 30  # Simpan histori 30 frame ke belakang (~1 detik)
OUTPUT_FOLDER = "hasil_kecelakaan_raw"
os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Folder untuk gambar mentah

def hitung_iou(boxA, boxB):
    """Menghitung seberapa tumpang-tindih dua kotak (IoU)"""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def pantau_kecelakaan(frame, frame_idx, detections, frame_buffer, box_history, id_terlibat_kecelakaan, width, height):
    """
    Menyimpan frame ke buffer dan mengecek tabrakan (IoU > 10%).
    Jika tabrakan, mundur ke masa lalu untuk crop gambar sebelum benturan.
    """
    kecelakaan_frame_ini = False
    tracker_ids = detections.tracker_id
    bboxes = detections.xyxy

    if tracker_ids is None or len(tracker_ids) == 0:
        return False

    # 1. Simpan frame utuh ke memori "Mesin Waktu"
    frame_buffer[frame_idx] = frame.copy()
    if frame_idx - BUFFER_SIZE in frame_buffer:
        del frame_buffer[frame_idx - BUFFER_SIZE]

    # 2. Simpan histori koordinat per ID kendaraan
    for i, t_id in enumerate(tracker_ids):
        box_history[t_id].append((frame_idx, bboxes[i]))
        if len(box_history[t_id]) > BUFFER_SIZE:
            box_history[t_id].pop(0)

    # LOGIKA MENCARI KECELAKAAN (IoU > 10%)
    if len(bboxes) > 1:
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = hitung_iou(bboxes[i], bboxes[j])

                if iou > 0.10:
                    kecelakaan_terjadi_di_frame_ini = True
                    kendaraan_terlibat = [tracker_ids[i], tracker_ids[j]]

                    for id_target in kendaraan_terlibat:
                        if id_target not in id_terlibat_kecelakaan:
                            id_terlibat_kecelakaan.append(id_target)

                            # MUNDUR KE MASA LALU
                            histori_kendaraan = box_history[id_target]
                            frame_lama_idx, bbox_lama = histori_kendaraan[0]

                            if frame_lama_idx in frame_buffer:
                                frame_sebelum_tabrak = frame_buffer[frame_lama_idx]

                                # EKSTRAK KOORDINAT LAMA & TAMBAH PADDING (MARGIN)
                                x1, y1, x2, y2 = map(int, bbox_lama)
                                padding = 30 # Lebarkan kotak 30 pixel
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(width, x2 + padding)
                                y2 = min(height, y2 + padding)

                                # Potong gambar
                                crop_img = frame_sebelum_tabrak[y1:y2, x1:x2]

                                # CEK VALIDITAS UKURAN
                                tinggi, lebar = crop_img.shape[:2]
                                if tinggi > 10 and lebar > 10:
                                    nama_file = f"{OUTPUT_FOLDER}/Plat_Raw_ID_{id_target}.jpg"
                                    try:
                                        cv2.imwrite(nama_file, crop_img)
                                        print(f"💥 KECELAKAAN! Tangkap ID {id_target} di frame {frame_lama_idx}. Tersimpan: {nama_file}")
                                    except Exception as e:
                                        print("Save error:", e)
                                

    return kecelakaan_frame_ini