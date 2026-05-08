# ============================================================
# UTS MK612 - Robotik dan Kecerdasan Buatan
# Nama      : Nicholas Nusi Pelmelay
# NIM       : 4212311028
# Soal 2    : Program Fuzzy Mamdani - Sistem Monitoring Laut
#             Uji: Ketinggian Air = 78 cm, Kecepatan Angin = 8.2 m/s
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

FOLDER = "hasil_fuzzy_nicholas"
os.makedirs(FOLDER, exist_ok=True)
# ══════════════════════════════════════════════════════════════
# SOAL 2: PROGRAM FUZZY
# ══════════════════════════════════════════════════════════════

# ── FUNGSI KEANGGOTAAN (MEMBERSHIP FUNCTIONS) ─────────────────

def mf_trimf(x, a, b, c):
    """Segitiga: naik dari a ke b, turun dari b ke c."""
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    elif x == b:
        return 1.0
    else:
        return (c - x) / (c - b)

def mf_trapmf_left(x, b, c):
    """Trapesium kiri (shoulder kiri): 1 untuk x<=b, turun ke 0 di c."""
    if x <= b:
        return 1.0
    elif x < c:
        return (c - x) / (c - b)
    else:
        return 0.0

def mf_trapmf_right(x, a, b):
    """Trapesium kanan (shoulder kanan): naik dari a ke b, lalu 1."""
    if x >= b:
        return 1.0
    elif x > a:
        return (x - a) / (b - a)
    else:
        return 0.0


# ── INPUT 1: KETINGGIAN AIR LAUT (cm) ─────────────────────────
# Surut  : shoulder kiri,  puncak di 0,   turun ke 0 di 80
# Normal : segitiga,       naik dari 60,  puncak 100, turun ke 140
# Pasang : shoulder kanan, naik dari 120, puncak di 200+

def ketinggian_surut(x):
    return mf_trapmf_left(x, 60, 80)

def ketinggian_normal(x):
    return mf_trimf(x, 60, 100, 140)

def ketinggian_pasang(x):
    return mf_trapmf_right(x, 120, 160)


# ── INPUT 2: KECEPATAN ANGIN (m/s) ────────────────────────────
# Tenang  : shoulder kiri,  turun ke 0 di 5
# Sedang  : segitiga,       naik dari 3, puncak 10, turun ke 17
# Kencang : shoulder kanan, naik dari 15+

def angin_tenang(x):
    return mf_trapmf_left(x, 3, 5)

def angin_sedang(x):
    return mf_trimf(x, 3, 10, 17)

def angin_kencang(x):
    return mf_trapmf_right(x, 15, 20)


# ── OUTPUT: STATUS LAUT ────────────────────────────────────────
# Aman     : shoulder kiri,  turun ke 0 di 30
# Waspada  : segitiga,       puncak 45
# Bahaya   : shoulder kanan, naik dari 60+

def status_aman(x):
    return mf_trapmf_left(x, 20, 30)

def status_waspada(x):
    return mf_trimf(x, 25, 45, 65)

def status_bahaya(x):
    return mf_trapmf_right(x, 60, 75)


# ── RULE BASE (Fuzzy Rules) ────────────────────────────────────
# Kombinasi logis berdasarkan studi kasus:
#
# R1: IF Surut   AND Tenang  → Aman
# R2: IF Surut   AND Sedang  → Aman
# R3: IF Surut   AND Kencang → Waspada
# R4: IF Normal  AND Tenang  → Aman
# R5: IF Normal  AND Sedang  → Waspada
# R6: IF Normal  AND Kencang → Bahaya
# R7: IF Pasang  AND Tenang  → Waspada
# R8: IF Pasang  AND Sedang  → Bahaya
# R9: IF Pasang  AND Kencang → Bahaya

def evaluasi_rules(tinggi, angin):
    """Hitung firing strength setiap rule dengan operator AND (min)."""
    s  = ketinggian_surut(tinggi)
    n  = ketinggian_normal(tinggi)
    p  = ketinggian_pasang(tinggi)
    te = angin_tenang(angin)
    se = angin_sedang(angin)
    ke = angin_kencang(angin)

    rules = {
        "R1 (Surut  & Tenang  → Aman)"    : ("aman",    min(s, te)),
        "R2 (Surut  & Sedang  → Aman)"    : ("aman",    min(s, se)),
        "R3 (Surut  & Kencang → Waspada)" : ("waspada", min(s, ke)),
        "R4 (Normal & Tenang  → Aman)"    : ("aman",    min(n, te)),
        "R5 (Normal & Sedang  → Waspada)" : ("waspada", min(n, se)),
        "R6 (Normal & Kencang → Bahaya)"  : ("bahaya",  min(n, ke)),
        "R7 (Pasang & Tenang  → Waspada)" : ("waspada", min(p, te)),
        "R8 (Pasang & Sedang  → Bahaya)"  : ("bahaya",  min(p, se)),
        "R9 (Pasang & Kencang → Bahaya)"  : ("bahaya",  min(p, ke)),
    }
    return rules


# ── AGREGASI & DEFUZZIFIKASI (Centroid / Center of Gravity) ───

def defuzzifikasi(rules, resolusi=1000):
    """
    Mamdani aggregation: clipping setiap MF output dengan alpha-cut,
    gabungkan dengan max, lalu defuzzifikasi dengan metode centroid.
    """
    x_out = np.linspace(0, 100, resolusi)
    agregasi = np.zeros(resolusi)

    for nama_rule, (konsekuen, alpha) in rules.items():
        for i, x in enumerate(x_out):
            if konsekuen == "aman":
                mu = status_aman(x)
            elif konsekuen == "waspada":
                mu = status_waspada(x)
            else:
                mu = status_bahaya(x)
            # Clipping: potong MF di nilai alpha
            agregasi[i] = max(agregasi[i], min(alpha, mu))

    # Centroid
    if np.sum(agregasi) == 0:
        return 0.0
    crisp = np.sum(x_out * agregasi) / np.sum(agregasi)
    return crisp, x_out, agregasi


def klasifikasi_status(nilai):
    if nilai < 30:
        return "AMAN"
    elif nilai < 60:
        return "WASPADA"
    else:
        return "BAHAYA"


# ── FUNGSI UTAMA INFERENSI ─────────────────────────────────────

def inferensi_fuzzy(tinggi_air, kec_angin, tampilkan=True):
    rules = evaluasi_rules(tinggi_air, kec_angin)
    hasil_defuzz = defuzzifikasi(rules)
    nilai_crisp, x_out, agregasi = hasil_defuzz
    status = klasifikasi_status(nilai_crisp)

    if tampilkan:
        print(f"\n  Input  → Ketinggian Air: {tinggi_air} cm | Kecepatan Angin: {kec_angin} m/s")
        print(f"  Output → Nilai Crisp: {nilai_crisp:.4f} | Status: {status}")
        print()
        print("  Detail Firing Strength setiap Rule:")
        for nama, (konsekuen, alpha) in rules.items():
            print(f"    {nama:45s} α = {alpha:.4f}")

    return nilai_crisp, status, rules, x_out, agregasi


# ══════════════════════════════════════════════════════════════
# SOAL 3: PERHITUNGAN MANUAL FUZZY MAMDANI
# Input: Ketinggian Air = 78 cm, Kecepatan Angin = 8.2 m/s
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# VISUALISASI
# ══════════════════════════════════════════════════════════════

def plot_membership_functions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Fungsi Keanggotaan - Sistem Fuzzy Monitoring Laut\n"
        "Nicholas Nusi Pelmelay (4212311028)",
        fontsize=13, fontweight="bold"
    )

    # Input 1: Ketinggian Air
    x1 = np.linspace(0, 220, 1000)
    axes[0].plot(x1, [ketinggian_surut(v) for v in x1],  "b-",  lw=2, label="Surut")
    axes[0].plot(x1, [ketinggian_normal(v) for v in x1], "g-",  lw=2, label="Normal")
    axes[0].plot(x1, [ketinggian_pasang(v) for v in x1], "r-",  lw=2, label="Pasang")
    axes[0].axvline(78, color="k", linestyle="--", lw=1.2, label="Input = 78 cm")
    axes[0].set_title("Input 1: Ketinggian Air Laut (cm)")
    axes[0].set_xlabel("Ketinggian (cm)")
    axes[0].set_ylabel("Derajat Keanggotaan")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.15)

    # Input 2: Kecepatan Angin
    x2 = np.linspace(0, 25, 1000)
    axes[1].plot(x2, [angin_tenang(v) for v in x2],  "b-",  lw=2, label="Tenang")
    axes[1].plot(x2, [angin_sedang(v) for v in x2],  "g-",  lw=2, label="Sedang")
    axes[1].plot(x2, [angin_kencang(v) for v in x2], "r-",  lw=2, label="Kencang")
    axes[1].axvline(8.2, color="k", linestyle="--", lw=1.2, label="Input = 8.2 m/s")
    axes[1].set_title("Input 2: Kecepatan Angin (m/s)")
    axes[1].set_xlabel("Kecepatan (m/s)")
    axes[1].set_ylabel("Derajat Keanggotaan")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.15)

    # Output: Status Laut
    x3 = np.linspace(0, 100, 1000)
    axes[2].plot(x3, [status_aman(v) for v in x3],    "b-",  lw=2, label="Aman")
    axes[2].plot(x3, [status_waspada(v) for v in x3], "g-",  lw=2, label="Waspada")
    axes[2].plot(x3, [status_bahaya(v) for v in x3],  "r-",  lw=2, label="Bahaya")
    axes[2].set_title("Output: Status Laut")
    axes[2].set_xlabel("Status (0–100)")
    axes[2].set_ylabel("Derajat Keanggotaan")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.05, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "membership_functions.png"), dpi=200)
    plt.close()
    print("\n    Grafik MF disimpan.")


def plot_hasil_inferensi(x_out, agregasi, nilai_crisp, status, tinggi, angin):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_out, agregasi, alpha=0.4, color="steelblue", label="Area agregasi")
    ax.plot(x_out, agregasi, color="steelblue", lw=2)
    ax.axvline(nilai_crisp, color="red", lw=2, linestyle="--",
               label=f"Centroid = {nilai_crisp:.2f} → {status}")
    ax.set_title(
        f"Hasil Inferensi Fuzzy Mamdani\n"
        f"Ketinggian Air = {tinggi} cm | Kecepatan Angin = {angin} m/s\n"
        f"Nicholas Nusi Pelmelay (4212311028)",
        fontsize=11
    )
    ax.set_xlabel("Status Laut (0=Aman ... 100=Bahaya)")
    ax.set_ylabel("Derajat Keanggotaan (agregasi)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "hasil_inferensi.png"), dpi=200)
    plt.close()
    print("    Grafik hasil inferensi disimpan.")


def plot_pengujian_berbagai_input():
    """Heatmap status laut untuk berbagai kombinasi input."""
    tinggi_range = np.linspace(0, 200, 50)
    angin_range  = np.linspace(0, 25, 50)
    hasil = np.zeros((len(angin_range), len(tinggi_range)))

    for i, a in enumerate(angin_range):
        for j, t in enumerate(tinggi_range):
            rules = evaluasi_rules(t, a)
            crisp, _, _ = defuzzifikasi(rules)
            hasil[i, j] = crisp

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.contourf(tinggi_range, angin_range, hasil, levels=20, cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Status Laut (0=Aman, 100=Bahaya)")
    ax.contour(tinggi_range, angin_range, hasil, levels=[30, 60],
               colors=["yellow", "red"], linewidths=2)
    ax.plot(78, 8.2, "k*", markersize=14, label="Uji Soal 3 (78 cm, 8.2 m/s)")
    ax.set_xlabel("Ketinggian Air Laut (cm)", fontsize=11)
    ax.set_ylabel("Kecepatan Angin (m/s)", fontsize=11)
    ax.set_title(
        "Peta Status Laut - Semua Kombinasi Input\n"
        "Nicholas Nusi Pelmelay (4212311028)\n"
        "(Garis kuning = batas Aman/Waspada | Garis merah = batas Waspada/Bahaya)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "peta_status_laut.png"), dpi=200)
    plt.close()
    print("    Peta status laut disimpan.")


# ══════════════════════════════════════════════════════════════
# MAIN — JALANKAN SEMUA
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  UTS MK612 - FUZZY MAMDANI - Nicholas Nusi Pelmelay")
    print("=" * 65)

    # ── SOAL 2: Uji beberapa skenario ────────────────────────
    print("\n[SOAL 2] PENGUJIAN SISTEM FUZZY MAMDANI")
    print("-" * 65)
    skenario = [
        (78,  8.2,  "Soal 3 (kasus utama)"),
        (50,  3.0,  "Surut & Angin tenang"),
        (100, 10.0, "Normal & Angin sedang"),
        (170, 18.0, "Pasang & Angin kencang"),
        (130, 12.0, "Pasang & Angin sedang"),
    ]
    print(f"\n  {'No':<4} {'Tinggi':>8} {'Angin':>8} {'Crisp':>8} {'Status':<12} Keterangan")
    print(f"  {'-'*70}")
    for i, (t, a, ket) in enumerate(skenario, 1):
        crisp, status, _, _, _ = inferensi_fuzzy(t, a, tampilkan=False)
        print(f"  {i:<4} {t:>7} cm {a:>6} m/s {crisp:>8.2f} {status:<12} {ket}")

    # ── Visualisasi ─────────────────────────────────────────
    print("\n[VISUALISASI] Menyimpan grafik...")
    plot_membership_functions()
    plot_pengujian_berbagai_input()

    print(f"\n{'=' * 65}")
    print(f"  Selesai. Semua file tersimpan di folder: '{FOLDER}/'")
    print(f"{'=' * 65}")
