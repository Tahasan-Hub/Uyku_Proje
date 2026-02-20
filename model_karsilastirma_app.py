import os
import time
import pygame
import math
import json
import logging
import glob
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import mediapipe as mp

from utils.core_logic import (
    CentroidTracker, ViolationManager, ViolationEpisode,
    eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX, is_point_in_rect
)

# ===========================================================
# CONFIGURATION LOAD
# ===========================================================
try:
    with open("config.json", "r") as f:
        ayarlar = json.load(f)
except Exception as e:
    st.error(f"config.json okunamadı: {e}")
    st.stop()

# ===========================================================
# LOGGING SETUP
# ===========================================================
LOG_DIR = ayarlar.get("log_dir", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"Log_{datetime.now().strftime('%Y-%m-%d')}.log")

logger = logging.getLogger("UykuTakipApp")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_format = logging.Formatter('%(asctime)s [%(levelname)s]    %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

# ===========================================================
# SABİTLER
# ===========================================================
MODEL_CONFIGS = ayarlar['model_configs']
DEVICE = ayarlar.get("device", 0)
CONF_THRESH = ayarlar.get("conf_threshold", 0.4)
IOU_TRACK_THRESH = ayarlar.get("iou_track_threshold", 100)
STILLNESS_SECONDS = ayarlar.get("stillness_seconds", 10.0)
EYE_CLOSED_SECONDS = ayarlar.get("eye_closed_seconds", 10.0)
MOVEMENT_PIXEL_THRESHOLD = ayarlar.get("movement_pixel_threshold", 20.0)
EAR_THRESHOLD = ayarlar.get("ear_threshold", 0.21)
OUTPUT_REPORT_DIR = ayarlar.get("output_report_dir", "raporlar")

# ===========================================================
# SES VE MEDIAPIPE
# ===========================================================
if not pygame.mixer.get_init():
    pygame.mixer.init()

try:
    uyari_sesi = pygame.mixer.Sound(ayarlar['alarm_settings']['uyari_sesi'])
    acil_sesi = pygame.mixer.Sound(ayarlar['alarm_settings']['acil_durum_sesi'])
except:
    uyari_sesi = acil_sesi = None

mp_face_mesh = mp.solutions.face_mesh

# ===========================================================
# GÖREV 5: BÖLGE SEÇİMİ (ZONE MONITORING)
# ===========================================================

def select_zones_from_video(video_path):
    """Videonun ilk karesinden ROI seçimi yaptırır."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return []
    
    window_name = "BOLGE SECIMI (Cizdikten sonra ENTER'a bas, bitince ESC'ye bas)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Pencereyi en üste getirmeye çalış (Windows için)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    st.warning("⚠️ LÜTFEN DİKKAT: Bölge seçimi penceresi tarayıcının arkasında kalmış olabilir. Görev çubuğundaki (taskbar) Python ikonuna tıklayarak pencereyi açın.")
    
    # Çoklu ROI seçimi
    # r tuşu ile çizim yapılır, Enter ile onaylanır. İşlem bitince ESC veya pencereyi kapatma.
    zones = cv2.selectROIs(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    formatted_zones = []
    if len(zones) > 0:
        for z in zones:
            x, y, w, h = z
            if w > 0 and h > 0: # Sadece geçerli seçimleri al
                formatted_zones.append((x, y, x+w, y+h))
    return formatted_zones

# ===========================================================
# ANALİZ ÇEKİRDEĞİ
# ===========================================================

def compute_ear_for_faces(frame, face_landmarks_list, track_boxes, violation_manager, current_sec):
    h, w, _ = frame.shape
    for face_landmarks in face_landmarks_list:
        coords = []
        for lm in face_landmarks.landmark:
            coords.append((int(lm.x * w), int(lm.y * h)))
        
        left_ear = eye_aspect_ratio([coords[i] for i in LEFT_EYE_IDX])
        right_ear = eye_aspect_ratio([coords[i] for i in RIGHT_EYE_IDX])
        ear = (left_ear + right_ear) / 2.0
        
        nose_x, nose_y = coords[NOSE_IDX]
        assigned_track_id = None
        for tid, bbox in track_boxes.items():
            if bbox[0] <= nose_x <= bbox[2] and bbox[1] <= nose_y <= bbox[3]:
                assigned_track_id = tid
                break
        
        if assigned_track_id is not None:
            violation_manager.update_eye_state(assigned_track_id, ear, current_sec)

def analyze_video_with_model(video_path, model_key, monitoring_zones=None):
    cfg = MODEL_CONFIGS[model_key]
    model = YOLO(cfg["engine"]) if os.path.exists(cfg["engine"]) else YOLO(cfg["pt"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return [], 0.0

    video_placeholder = st.empty()
    tracker = CentroidTracker(max_distance=IOU_TRACK_THRESH)
    violation_manager = ViolationManager(
        still_threshold=STILLNESS_SECONDS, eye_threshold=EYE_CLOSED_SECONDS,
        movement_threshold=MOVEMENT_PIXEL_THRESHOLD, ear_threshold=EAR_THRESHOLD
    )
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    tracked_persons = set()
    log_flags = {"warning": set(), "critical": set()}
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    total_time_processing = 0.0
    processed_frames = 0
    progress_bar = st.progress(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_sec = frame_index / fps_video if fps_video > 0 else 0.0
            frame_index += 1
            start_time = time.time()

            # YOLO Tespiti
            yolo_results = model(frame, conf=CONF_THRESH, classes=[0], device=DEVICE, verbose=False)
            raw_detections = [tuple(map(int, box.xyxy[0].tolist())) for box in yolo_results[0].boxes] if yolo_results[0].boxes is not None else []
            
            # GÖREV 5: BÖLGE FİLTRELEME
            detections = []
            if monitoring_zones:
                for det in raw_detections:
                    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
                    if any(is_point_in_rect((cx, cy), zone) for zone in monitoring_zones):
                        detections.append(det)
            else:
                detections = raw_detections

            track_boxes = tracker.update(detections)
            for tid in track_boxes:
                if tid not in tracked_persons:
                    logger.info(f"Kisi tespit edildi: ID={tid}")
                    tracked_persons.add(tid)
            
            violation_manager.update_tracks(track_boxes, current_sec)
            face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face_results.multi_face_landmarks:
                compute_ear_for_faces(frame, face_results.multi_face_landmarks, track_boxes, violation_manager, current_sec)

            global_violation, reasons, per_person_timers = violation_manager.compute_violations(current_sec)

            # Görselleştirme: Bölgeleri çiz
            if monitoring_zones:
                for i, zone in enumerate(monitoring_zones):
                    cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), 2)
                    cv2.putText(frame, f"Bolge {i+1}", (zone[0], zone[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if global_violation:
                alarm_tetikle("acil" if "Goz Kapali" in reasons else "uyari")
                color = (0, 0, 255) if "Goz Kapali" in reasons else (0, 255, 255)
                cv2.putText(frame, f"IHLAL: {' / '.join(reasons)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                pygame.mixer.stop()

            # UI Update
            total_time_processing += (time.time() - start_time)
            processed_frames += 1
            if total_frames > 0: progress_bar.progress(min(1.0, frame_index / total_frames))
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        violation_manager.finalize(current_sec)
    finally:
        cap.release()
        face_mesh.close()
        progress_bar.empty()

    avg_fps = processed_frames / total_time_processing if total_time_processing > 0 else 0.0
    return violation_manager.episodes, avg_fps

# ===========================================================
# ALARM VE RAPOR
# ===========================================================

son_alarm_zamani = 0
def alarm_tetikle(tur):
    global son_alarm_zamani
    su_an = time.time()
    if ayarlar['alarm_settings']['enabled'] and (su_an - son_alarm_zamani > ayarlar['alarm_settings']['cooldown_seconds']):
        pygame.mixer.stop()
        if tur == "acil" and acil_sesi: acil_sesi.play()
        elif uyari_sesi: uyari_sesi.play()
        son_alarm_zamani = su_an

def save_report_csv(model_results):
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    fpath = os.path.join(OUTPUT_REPORT_DIR, f"Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(fpath, "w", newline='', encoding="utf-8-sig") as f:
        f.write("model;ihlal_turu;baslangic;bitis;sure\n")
        for mk, data in model_results.items():
            for ep in data["episodes"]:
                f.write(f"{MODEL_CONFIGS[mk]['label']};{ep.violation_type};{ep.start_sec:.2f};{ep.end_sec:.2f};{ep.duration:.2f}\n")
    return fpath

# ===========================================================
# GÖREV 6: GÜNLÜK ÖZET DASHBOARD
# ===========================================================

def render_dashboard():
    st.subheader("📊 Günlük Özet & Analitik Dashboard")
    report_files = glob.glob(os.path.join(OUTPUT_REPORT_DIR, "Rapor_*.csv"))
    
    if not report_files:
        st.warning("Henüz analiz verisi bulunamadı. Lütfen önce bir video analizi yapın.")
        return

    df_list = []
    for f in report_files:
        try:
            temp_df = pd.read_csv(f, sep=';', encoding='utf-8-sig')
            fname = os.path.basename(f)
            date_str = fname.split("_")[1]
            temp_df['tarih'] = pd.to_datetime(date_str, format='%Y%m%d')
            temp_df['saat'] = int(fname.split("_")[2][:2]) # Dosya adından saat bilgisini al
            df_list.append(temp_df)
        except: continue
    
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    
    # Metrikler
    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam İhlal Sayısı", len(df))
    m2.metric("Ortalama İhlal Süresi", f"{df['sure'].mean():.2f} sn")
    peak_hour = df['saat'].mode()[0] if not df.empty else 0
    m3.metric("En Yoğun Saat Dilimi", f"{peak_hour}:00 - {peak_hour+1}:00")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("🕒 **Saatlik İhlal Dağılımı**")
        hourly_df = df.groupby('saat').size().reset_index(name='sayi')
        fig_hour = px.bar(hourly_df, x='saat', y='sayi', labels={'saat':'Saat', 'sayi':'İhlal Sayısı'}, color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        st.write("📈 **Günlük İhlal Trendi**")
        trend_df = df.groupby('tarih').size().reset_index(name='sayi')
        fig_trend = px.line(trend_df, x='tarih', y='sayi', labels={'tarih':'Tarih', 'sayi':'İhlal'}, markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.write("🤖 **Model Karsılaştırması**")
    model_df = df.groupby(['model', 'ihlal_turu']).size().reset_index(name='sayi')
    fig_model = px.bar(model_df, x='model', y='sayi', color='ihlal_turu', barmode='group')
    st.plotly_chart(fig_model, use_container_width=True)

# ===========================================================
# ARAYÜZ (MAIN)
# ===========================================================

def main():
    st.set_page_config(page_title="AI Uyku Analiz Sistemi", layout="wide", page_icon="🌙")
    st.title("🌙 AI Uyku ve Güvenlik Takip Sistemi")
    
    tab1, tab2 = st.tabs(["🎥 Video Analizi", "📊 Analitik Dashboard"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("🛠️ Kontrol Paneli")
            ayarlar['alarm_settings']['enabled'] = st.toggle("🚨 Alarm Sesleri", value=ayarlar['alarm_settings']['enabled'])
            use_roi = st.toggle("🎯 Bölge İzleme (ROI) Aktif", value=False)
            selected_models = st.multiselect("Modeller", options=list(MODEL_CONFIGS.keys()), default=[list(MODEL_CONFIGS.keys())[0]], format_func=lambda k: MODEL_CONFIGS[k]["label"])
            uploaded_file = st.file_uploader("📂 Video Dosyası", type=["mp4"])
            start_btn = st.button("🚀 Analizi Başlat")

        with col2:
            if not uploaded_file: st.info("Video yükleyerek analize başlayın.")
            elif start_btn:
                temp_path = f"temp_video.mp4"
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                # ROI SEÇİMİ (GÖREV 5)
                monitoring_zones = select_zones_from_video(temp_path) if use_roi else None
                
                results = {}
                for m_key in selected_models:
                    st.subheader(f"🔍 {MODEL_CONFIGS[m_key]['label']} Analiz Ediliyor...")
                    episodes, fps = analyze_video_with_model(temp_path, m_key, monitoring_zones=monitoring_zones)
                    results[m_key] = {"episodes": episodes, "fps": fps}
                
                save_report_csv(results)
                st.success("Analiz tamamlandı. Dashboard sekmesinden detayları görebilirsiniz.")
                if os.path.exists(temp_path): os.remove(temp_path)

    with tab2:
        render_dashboard()

if __name__ == "__main__":
    main()
