# fitur_counting.py
from collections import defaultdict


class VehicleCounter:
    """Menghitung kendaraan yang melintasi garis virtual.

    Setiap kendaraan (tracker_id) dihitung TEPAT SATU KALI pada saat pertama
    kali melintasi salah satu garis, sehingga tidak terjadi double-count
    saat kendaraan melewati kedua garis. Arah dicatat sebagai:

    - "in"  : masuk ke segmen (dari atas melewati Garis1, atau dari bawah melewati Garis2)
    - "out" : keluar dari segmen (ke atas melewati Garis1, atau ke bawah melewati Garis2)
    """

    def __init__(self):
        self.timestamp_l1 = {}          # tid -> waktu pertama lintasi Garis 1
        self.timestamp_l2 = {}          # tid -> waktu pertama lintasi Garis 2
        self.class_counts = defaultdict(int)
        self.in_count = 0
        self.out_count = 0
        self._counted = {}              # tid -> waktu saat dihitung (untuk cleanup RAM)

    def update(self, tid, cid, c1_in, c1_out, c2_in, c2_out, t_sekarang):
        """Update counter dengan status lintas per kendaraan pada satu frame.

        c1_in/c1_out/c2_in/c2_out adalah nilai boolean dari LineZone.trigger
        untuk kendaraan tersebut (indeks yang sama dengan deteksi).
        """
        if tid is None:
            return

        if c1_in or c1_out:
            self.timestamp_l1.setdefault(tid, t_sekarang)
        if c2_in or c2_out:
            self.timestamp_l2.setdefault(tid, t_sekarang)

        if tid in self._counted:
            return

        if c1_in:            # masuk dari atas melewati Garis 1
            self._count(tid, cid, "in", t_sekarang)
        elif c1_out:         # keluar ke atas melewati Garis 1
            self._count(tid, cid, "out", t_sekarang)
        elif c2_out:         # masuk dari bawah melewati Garis 2
            self._count(tid, cid, "in", t_sekarang)
        elif c2_in:          # keluar ke bawah melewati Garis 2
            self._count(tid, cid, "out", t_sekarang)

    def _count(self, tid, cid, arah, t_sekarang):
        self._counted[tid] = t_sekarang
        self.class_counts[cid] += 1
        if arah == "in":
            self.in_count += 1
        else:
            self.out_count += 1

    def get_total_akumulasi(self):
        """Total semua kendaraan yang sudah terhitung pada periode berjalan."""
        return sum(self.class_counts.values())

    def get_in_out(self):
        return self.in_count, self.out_count

    def get_per_kelas(self):
        return dict(self.class_counts)

    def bersihkan_memori(self, t_sekarang, timeout=30.0):
        """Hapus ingatan kendaraan yang sudah lewat > timeout agar RAM tidak penuh."""
        stale1 = [k for k, v in self.timestamp_l1.items() if t_sekarang - v > timeout]
        stale2 = [k for k, v in self.timestamp_l2.items() if t_sekarang - v > timeout]
        for k in stale1:
            del self.timestamp_l1[k]
        for k in stale2:
            del self.timestamp_l2[k]

        stale_counted = [k for k, v in self._counted.items() if t_sekarang - v > timeout]
        for k in stale_counted:
            del self._counted[k]
