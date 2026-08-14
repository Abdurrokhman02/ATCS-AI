# fitur_counting.py

def catat_dan_hitung(tid, cid, crossed_l1_i, crossed_l2_i, t_sekarang, timestamp_l1, timestamp_l2, class_counts):
    """Mencatat waktu objek melintasi garis dan menambah jumlah kendaraan."""
    if crossed_l1_i and tid not in timestamp_l1:
        timestamp_l1[tid] = t_sekarang
        class_counts[cid] += 1
        
    if crossed_l2_i and tid not in timestamp_l2:
        timestamp_l2[tid] = t_sekarang
        # Jangan dihitung lagi kalau dia cuma nyebrang dari Garis 1
        if not crossed_l1_i: 
            class_counts[cid] += 1

def get_total_akumulasi(class_counts):
    """Mengembalikan total semua kendaraan yang sudah lewat."""
    return sum(class_counts.values())

def bersihkan_memori(timestamp_l1, timestamp_l2, speeds_list, travel_times, t_sekarang):
    """Hapus ingatan kendaraan yang sudah lewat > 10 detik agar RAM tidak penuh."""
    keys_to_del_1 = [k for k, v in timestamp_l1.items() if t_sekarang - v > 10]
    keys_to_del_2 = [k for k, v in timestamp_l2.items() if t_sekarang - v > 10]
    
    for k in keys_to_del_1: del timestamp_l1[k]
    for k in keys_to_del_2: del timestamp_l2[k]
    
    # Potong list jika terlalu panjang (menyisakan 50 data terbaru)
    if len(speeds_list) > 100: speeds_list[:] = speeds_list[-50:]
    if len(travel_times) > 100: travel_times[:] = travel_times[-50:]