# ============================================================
# UTS MK612 - Robotik dan Kecerdasan Buatan
# Nama      : Nicholas Nusi Pelmelay
# NIM       : 4212311028
# Kelas     : Malam A
# Metode    : Decision Tree
# Parameter : 345 (fBodyAccJerk-mean()-X)
#             346 (fBodyAccJerk-mean()-Y)
#             347 (fBodyAccJerk-mean()-Z)
# Target    : Activity
# Dataset   : Data UTS.csv (7352 baris, 563 kolom)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# Buat folder output
FOLDER = "hasil_nicholas"
os.makedirs(FOLDER, exist_ok=True)


# ── SOAL 1: PREPROCESSING DATA (TP-2, TP-3) ──────────────────────────────────
# Jenis preprocessing yang digunakan:
#   1. Seleksi fitur (memilih parameter 345, 346, 347 sesuai pembagian kelompok)
#   2. Penghapusan data kosong (dropna)
#   3. Standarisasi fitur (StandardScaler) agar skala data seragam
#      → cocok untuk dataset sensor akselerometer yang memiliki rentang nilai kecil
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  UTS MK612 - Nicholas Nusi Pelmelay (4212311028)")
print("=" * 60)

# 1. MEMBACA DATASET
print("\n[1] Membaca dataset...")
df = pd.read_csv("Data UTS.csv", sep=";", header=[0, 1])
print(f"    Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")

# 2. SELEKSI FITUR DAN TARGET (akses via posisi kolom agar aman)
#    Parameter 345 = index 344, 346 = index 345, 347 = index 346 (0-based)
print("\n[2] Memilih fitur parameter 345, 346, 347...")
X = df.iloc[:, [344, 345, 346]].copy()
X.columns = [
    "fBodyAccJerk_mean_X",   # Parameter 345
    "fBodyAccJerk_mean_Y",   # Parameter 346
    "fBodyAccJerk_mean_Z",   # Parameter 347
]
y = df.iloc[:, -1].copy()
y.name = "Activity"

# 3. HAPUS DATA KOSONG
data = pd.concat([X, y], axis=1).dropna()
X = data[["fBodyAccJerk_mean_X", "fBodyAccJerk_mean_Y", "fBodyAccJerk_mean_Z"]]
y = data["Activity"]

print(f"    Data setelah dropna: {len(data)} baris")
print(f"    Data kosong ditemukan: {df.shape[0] - len(data)} baris")

# 4. STANDARISASI FITUR
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

print("\n    Statistik fitur SEBELUM standarisasi:")
print(X.describe().round(4).to_string())
print("\n    Statistik fitur SETELAH standarisasi (mean≈0, std≈1):")
print(X_scaled.describe().round(4).to_string())

# Distribusi kelas
print("\n    Distribusi kelas Activity:")
print(y.value_counts().to_string())


# ── SOAL 2: KLASIFIKASI DECISION TREE (TP-4, TP-5) ───────────────────────────

print("\n[3] Membagi data latih dan uji (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"    Data latih : {X_train.shape[0]} baris")
print(f"    Data uji   : {X_test.shape[0]} baris")

print("\n[4] Melatih model Decision Tree...")
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=7,
    random_state=42
)
model.fit(X_train, y_train)
print("    Model berhasil dilatih.")

print("\n[5] Evaluasi model...")
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  HASIL EVALUASI")
print("=" * 60)
print(f"  Akurasi: {round(akurasi * 100, 2)}%")
print()
print(classification_report(y_test, y_pred))

# Simpan classification report
report_text = classification_report(y_test, y_pred)
with open(os.path.join(FOLDER, "classification_report.txt"), "w") as f:
    f.write("HASIL KLASIFIKASI DECISION TREE\n")
    f.write("=" * 50 + "\n")
    f.write("Nama      : Nicholas Nusi Pelmelay\n")
    f.write("NIM       : 4212311028\n")
    f.write("Parameter : 345, 346, 347\n")
    f.write("Target    : Activity\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Akurasi: {round(akurasi * 100, 2)}%\n\n")
    f.write(report_text)
print("    Classification report disimpan.")


# ── SOAL 3: PLOT PENYEBARAN DATA ──────────────────────────────────────────────

print("\n[6] Membuat visualisasi...")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
plt.title("Confusion Matrix - Decision Tree\nNicholas Nusi Pelmelay (Parameter 345, 346, 347)")
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "confusion_matrix.png"), dpi=200)
plt.close()
print("    Confusion matrix disimpan.")

# --- Visualisasi Decision Tree ---
plt.figure(figsize=(28, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title("Decision Tree - Nicholas Nusi Pelmelay (Parameter 345, 346, 347)")
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "decision_tree.png"), dpi=200)
plt.close()
print("    Visualisasi decision tree disimpan.")

# --- 3D Scatter Plot Penyebaran Data ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

colors = plt.cm.tab10(np.linspace(0, 1, len(y.unique())))
for aktivitas, warna in zip(sorted(y.unique()), colors):
    subset = data[data["Activity"] == aktivitas]
    ax.scatter(
        subset["fBodyAccJerk_mean_X"],
        subset["fBodyAccJerk_mean_Y"],
        subset["fBodyAccJerk_mean_Z"],
        label=aktivitas,
        color=warna,
        alpha=0.5,
        s=10,
    )

ax.set_xlabel("Param 345: fBodyAccJerk-mean()-X", fontsize=9)
ax.set_ylabel("Param 346: fBodyAccJerk-mean()-Y", fontsize=9)
ax.set_zlabel("Param 347: fBodyAccJerk-mean()-Z", fontsize=9)
ax.set_title(
    "Plot Penyebaran Data - Parameter 345, 346, 347\n"
    "Nicholas Nusi Pelmelay (4212311028)",
    fontsize=11,
)
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "scatter_3D.png"), dpi=200)
plt.close()
print("    3D scatter plot disimpan.")

# --- 2D Scatter Plot (pasangan fitur) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
pairs = [
    ("fBodyAccJerk_mean_X", "fBodyAccJerk_mean_Y", "Param 345 vs 346"),
    ("fBodyAccJerk_mean_X", "fBodyAccJerk_mean_Z", "Param 345 vs 347"),
    ("fBodyAccJerk_mean_Y", "fBodyAccJerk_mean_Z", "Param 346 vs 347"),
]
for ax, (px, py, judul), warna in zip(axes, pairs, [colors] * 3):
    for aktivitas, c in zip(sorted(y.unique()), colors):
        subset = data[data["Activity"] == aktivitas]
        ax.scatter(subset[px], subset[py], label=aktivitas, color=c, alpha=0.4, s=8)
    ax.set_xlabel(px, fontsize=9)
    ax.set_ylabel(py, fontsize=9)
    ax.set_title(judul, fontsize=10)
    ax.legend(fontsize=7)

fig.suptitle(
    "Plot Penyebaran Data 2D - Nicholas Nusi Pelmelay (4212311028)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "scatter_2D.png"), dpi=200)
plt.close()
print("    2D scatter plot disimpan.")

print("\n" + "=" * 60)
print(f"  Selesai. Semua file tersimpan di folder: '{FOLDER}/'")
print("=" * 60)