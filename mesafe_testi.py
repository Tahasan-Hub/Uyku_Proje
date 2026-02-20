import os
import time
import json
import cv2
import logging
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.core_logic import (
    CentroidTracker, eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
)

# ===========================================================
# CONFIGURATION & LOGGING
# ===========================================================
CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()

# Klasör ayarları
REPORT_DIR = config.get("output_report_dir", "raporlar")
os.makedirs(REPORT_DIR, exist_ok=True)

# Model & Cihaz ayarları
selected_model_key = config.get("selected_model", "yolo11n")
model_info = config["model_configs"][selected_model_key]
MODEL_PATH = model_info["engine"] if os.path.exists(model_info["engine"]) else model_info["pt"]
DEVICE = config.get("device", 0)
CONF_THRESH = config.get("conf_threshold", 0.4)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh

# ===========================================================
# ÇEKİRDEK ANALİZ FONKSİYONU (PROGRESS BAR EKLENDİ)
# ===========================================================

def analyze_video_simulated(video_path, distance_meters):
    """
    Bir videoyu fiziksel mesafe-ölçek formülüne göre simüle eder.
    """
    if not os.path.exists(video_path):
        return None

    scale = 1.0 / distance_meters

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    model = YOLO(MODEL_PATH)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    person_detected_frames = 0
    ear_success_frames = 0
    start_time = time.time()

    # TQDM İLERLEME ÇUBUĞU (%) - İndirme gibi ilerleme çubuğu
    pbar = tqdm(total=total_frames, desc=f"  Analiz {distance_meters}m", unit="kare", leave=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]

        # SİMUASYON: Mesafe bazlı çözünürlük düşürme
        if distance_meters > 1.0:
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sim_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            sim_frame = frame

        # YOLO Tespiti
        results = model(sim_frame, conf=CONF_THRESH, classes=[0], device=DEVICE, verbose=False)
        person_boxes = results[0].boxes if results[0].boxes is not None else []
        
        if len(person_boxes) > 0:
            person_detected_frames += 1
            
            # EAR Başarısı
            rgb_frame = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                ear_success_frames += 1

        pbar.update(1) # Çubuğu % olarak ilerlet

    pbar.close() # Çubuğu bitir
    total_processing_time = time.time() - start_time
    cap.release()
    face_mesh.close()

    detection_rate = (person_detected_frames / total_frames) * 100
    ear_success_rate = (ear_success_frames / person_detected_frames * 100) if person_detected_frames > 0 else 0
    fps = total_frames / total_processing_time if total_processing_time > 0 else 0

    return {
        "distance": distance_meters,
        "total_frames": total_frames,
        "detected_frames": person_detected_frames,
        "ear_frames": ear_success_frames,
        "detection_rate": round(detection_rate, 2),
        "ear_success_rate": round(ear_success_rate, 2),
        "total_time": round(total_processing_time, 2),
        "fps": round(fps, 2)
    }

# ===========================================================
# GÖRSELLEŞTİRME FONKSİYONU
# ===========================================================

def generate_performance_plot(all_results, timestamp):
    """Mesafeye göre % Başarı grafiği oluşturur."""
    distances = [res['distance'] for res in all_results]
    det_rates = [res['detection_rate'] for res in all_results]
    ear_rates = [res['ear_success_rate'] for res in all_results]
    fps_values = [res['fps'] for res in all_results]

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # % BAŞARI GRAFİĞİ
    ax1.plot(distances, det_rates, marker='o', label='Kisi Tespit %', color='#3498db', linewidth=3)
    ax1.plot(distances, ear_rates, marker='s', label='EAR Basari %', color='#2ecc71', linewidth=3)
    ax1.set_title('Mesafe Bazlı % Başarı Analizi', fontsize=14, pad=15)
    ax1.set_xlabel('Mesafe (Metre)')
    ax1.set_ylabel('Başarı Oranı (%)')
    ax1.set_xticks(distances)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # FPS VE ZAMAN ANALİZİ
    ax2.bar(distances, fps_values, color='#e67e22', alpha=0.7, label='FPS Performansı')
    ax2.set_title('İşlem Hızı (FPS) Analizi', fontsize=14, pad=15)
    ax2.set_xlabel('Mesafe (Metre)')
    ax2.set_ylabel('Saniyedeki Kare (FPS)')
    ax2.set_xticks(distances)
    ax2.legend()

    plt.tight_layout()
    plot_file = os.path.join(REPORT_DIR, f"Mesafe_Performans_Grafigi_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    return plot_file

# ===========================================================
# MAIN ENTRY POINT
# ===========================================================

def main():
    print("\n" + "═"*85)
    print(" PROFESYONEL MESAFE BAZLI PERFORMANS ANALİZ SİSTEMİ ".center(85))
    print("═"*85 + "\n")

    source_video = "uyku_testi.mp4" 

    if not os.path.exists(source_video):
        print(f"❌ HATA: '{source_video}' kaynak dosyası bulunamadı!")
        return

    test_distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_results = []
    
    print(f"📂 Kaynak Video: {source_video}")
    print(f"🤖 Model: {MODEL_PATH} | ⚙️ Cihaz: {DEVICE}\n")

    for d in test_distances:
        metrics = analyze_video_simulated(source_video, d)
        if metrics:
            all_results.append(metrics)
            print(f"✅ {d}m Analiz Tamamlandı: Tespit %{metrics['detection_rate']} | EAR %{metrics['ear_success_rate']} ")
        else:
            print(f"❌ {d}m Analiz Başarısız! ")

    # Ekrana Tablo Basma
    print("\n" + "╔" + "═"*81 + "╗")
    print("║" + " TEST SONUÇLARI ÖZET TABLOSU ".center(81) + "║")
    print("╠" + "═"*8 + "╤" + "═"*18 + "╤" + "═"*17 + "╤" + "═"*15 + "╤" + "═"*17 + "╣")
    print(f"║ {'Mesafe':<6} │ {'İşlenen Kare':<16} │ {'Tespit Oranı':<15} │ {'EAR Başarı':<13} │ {'Hız (FPS)':<15} ║")
    print("╟" + "─"*8 + "┼" + "─"*18 + "┼" + "─"*17 + "┼" + "─"*15 + "┼" + "─"*17 + "╢")
    for res in all_results:
        kare_bilgisi = f"{res['detected_frames']}/{res['total_frames']}"
        print(f"║ {res['distance']:>5}m   │ {kare_bilgisi:<16} │ %{res['detection_rate']:<14} │ %{res['ear_success_rate']:<13} │ {res['fps']:<15} ║")
    print("╚" + "═"*8 + "╧" + "═"*18 + "╧" + "═"*17 + "╧" + "═"*15 + "╧" + "═"*17 + "╝")

    # Raporları Kaydetme
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = generate_performance_plot(all_results, timestamp)
    print(f"\n📊 Performans grafiği (PNG) kaydedildi: {plot_file}")

    report_file = os.path.join(REPORT_DIR, f"Mesafe_Performans_Raporu_{timestamp}.csv")
    try:
        with open(report_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter=';')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"📄 Detaylı CSV raporu kaydedildi: {report_file}")
    except Exception as e:
        print(f"⚠️ Rapor kaydedilemedi: {e}")

if __name__ == "__main__":
    main()
