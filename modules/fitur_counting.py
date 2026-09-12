from collections import defaultdict


class VehicleCounter:
    """
    Menghitung kendaraan yang benar-benar melewati SEGMENT di antara LINE1 dan LINE2.

    Kendaraan baru dihitung setelah melewati KEDUA garis.
    Urutan crossing menentukan arah perjalanan:

    - LINE1 -> LINE2 : atas ke bawah
    - LINE2 -> LINE1 : bawah ke atas

    IN/OUT:
    - IN  = kendaraan berhasil masuk dan melewati segment
    - OUT = kendaraan berhasil keluar dari segment

    Untuk setiap kendaraan yang berhasil melewati kedua garis:
        total kendaraan +1
        class_counts +1
        in_count +1
        out_count +1

    Crossing LineZone (c1_in/c1_out/c2_in/c2_out) tetap dipakai
    hanya untuk mengetahui bahwa kendaraan telah melewati garis.
    """

    def __init__(self, arah_lalu_lintas="atas_ke_bawah", counted_timeout=30.0):
        self.timestamp_l1 = {}
        self.timestamp_l2 = {}

        self.class_counts = defaultdict(int)

        self.in_count = 0
        self.out_count = 0

        # tid -> waktu kendaraan sudah menyelesaikan kedua garis
        self._counted = {}

        # tid -> informasi crossing pertama
        # {
        #     "line": 1 atau 2,
        #     "timestamp": float,
        #     "class_id": int
        # }
        self._pending = {}

        self.arah_lalu_lintas = arah_lalu_lintas
        self._counted_timeout = counted_timeout

    def update(
        self,
        tid,
        cid,
        c1_in,
        c1_out,
        c2_in,
        c2_out,
        t_sekarang,
    ):
        """
        Update counter berdasarkan crossing LINE1 dan LINE2.

        Kendaraan belum dihitung ketika baru melewati satu garis.
        Kendaraan baru dihitung setelah crossing garis kedua terdeteksi.
        """

        if tid is None:
            return

        # Simpan timestamp masing-masing garis untuk kebutuhan speed.
        if c1_in or c1_out:
            self.timestamp_l1.setdefault(tid, t_sekarang)

        if c2_in or c2_out:
            self.timestamp_l2.setdefault(tid, t_sekarang)

        # Kendaraan yang sudah menyelesaikan segment tidak dihitung lagi.
        if tid in self._counted:
            return

        crossed_l1 = c1_in or c1_out
        crossed_l2 = c2_in or c2_out

        # Tidak melewati garis apa pun pada frame ini.
        if not crossed_l1 and not crossed_l2:
            return

        # Tentukan garis yang baru dilewati.
        current_line = None

        if crossed_l1:
            current_line = 1

        if crossed_l2:
            # Jika dua garis ter-trigger pada frame yang sama,
            # belum ada informasi urutan yang valid.
            if current_line == 1:
                return
            current_line = 2

        # ---------------------------------------------------------
        # CROSSING PERTAMA
        # ---------------------------------------------------------
        if tid not in self._pending:
            self._pending[tid] = {
                "line": current_line,
                "timestamp": t_sekarang,
                "class_id": cid,
            }
            return

        first_line = self._pending[tid]["line"]

        # Crossing garis yang sama lagi bukan crossing kedua.
        if first_line == current_line:
            return

        # ---------------------------------------------------------
        # CROSSING KEDUA -> PERJALANAN SEGMEN SELESAI
        # ---------------------------------------------------------

        if first_line == 2 and current_line == 1:
            arah = "atas_ke_bawah"
        elif first_line == 1 and current_line == 2:
            arah = "bawah_ke_atas"
        else:
            return

        # Hitung kendaraan yang berhasil melewati kedua garis.
        self._complete_vehicle(
            tid=tid,
            cid=self._pending[tid]["class_id"],
            arah=arah,
            t_sekarang=t_sekarang,
        )

        # Hapus state pending.
        del self._pending[tid]

    def _complete_vehicle(self, tid, cid, arah, t_sekarang):
        """
        Tandai kendaraan sebagai selesai melewati kedua garis.
        """

        self._counted[tid] = t_sekarang

        # Satu kendaraan selesai = satu total kendaraan.
        self.class_counts[cid] += 1

        # Untuk flow segment:
        # satu kendaraan yang berhasil masuk + keluar menghasilkan
        # IN dan OUT masing-masing satu.
        self.in_count += 1
        self.out_count += 1

    def get_total_akumulasi(self):
        """Total kendaraan yang sudah berhasil melewati kedua garis."""
        return sum(self.class_counts.values())

    def get_in_out(self):
        return self.in_count, self.out_count

    def get_per_kelas(self):
        return dict(self.class_counts)

    def bersihkan_memori(self, t_sekarang, timeout=None):
        """
        Bersihkan state lama agar RAM tidak terus bertambah.

        Cleanup berlaku untuk:
        - timestamp_l1
        - timestamp_l2
        - _pending
        - _counted
        """

        if timeout is None:
            timeout = self._counted_timeout

        stale1 = [
            k
            for k, v in self.timestamp_l1.items()
            if t_sekarang - v > timeout
        ]

        stale2 = [
            k
            for k, v in self.timestamp_l2.items()
            if t_sekarang - v > timeout
        ]

        stale_pending = [
            k
            for k, v in self._pending.items()
            if t_sekarang - v["timestamp"] > timeout
        ]

        stale_counted = [
            k
            for k, v in self._counted.items()
            if t_sekarang - v > timeout
        ]

        for k in stale1:
            del self.timestamp_l1[k]

        for k in stale2:
            del self.timestamp_l2[k]

        for k in stale_pending:
            del self._pending[k]

        for k in stale_counted:
            del self._counted[k]