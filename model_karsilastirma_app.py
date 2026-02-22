import os
import time
import requests
import pygame
import math
import json
import logging
import glob
from datetime import datetime
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
    eye_aspect_ratio, calculate_head_drop_ratio, is_point_in_rect,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX, 
    POSE_NOSE_IDX, POSE_LEFT_SHOULDER_IDX, POSE_RIGHT_SHOULDER_IDX
)

# ===========================================================
# CONFIGURATION & LOGGING
# ===========================================================
try:
    with open("config.json", "r") as f:
        ayarlar = json.load(f)
except Exception as e:
    st.error(f"config.json okunamadı: {e}")
    st.stop()

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
STILLNESS_SECONDS = ayarlar.get("stillness_seconds", 2.0)
EYE_CLOSED_SECONDS = ayarlar.get("eye_closed_seconds", 2.0)
MOVEMENT_PIXEL_THRESHOLD = ayarlar.get("movement_pixel_threshold", 20.0)
EAR_THRESHOLD = ayarlar.get("ear_threshold", 0.21)
OUTPUT_REPORT_DIR = ayarlar.get("output_report_dir", "raporlar")

# ===========================================================
# MEDIAPIPE & SES
# ===========================================================
if not pygame.mixer.get_init():
    pygame.mixer.init()

try:
    uyari_sesi = pygame.mixer.Sound(ayarlar['alarm_settings']['uyari_sesi'])
    acil_sesi = pygame.mixer.Sound(ayarlar['alarm_settings']['acil_durum_sesi'])
except:
    uyari_sesi = acil_sesi = None

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ===========================================================
# TELEGRAM & SNAPSHOT
# ===========================================================

def telegram_foto_gonder(frame, mesaj):
    TOKEN = "8103916174:AAHjf12NMt0o7aCIfep7Wsn_F3rnACKQXZQ"
    Chat_ID = "6864092011"
    
    # KVKK: Telegram'a giden fotoğrafta yüzü zaten ana döngüde buladıysak o haliyle gider.
    _, buffer = cv2.imencode(".jpg", frame)
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    payload = {"chat_id": Chat_ID, "caption": mesaj}
    files = {"photo": ("guardwatch_alarm.jpg", buffer.tobytes(), "image/jpeg")}
    
    try:
        r = requests.post(url, data=payload, files=files, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram: Fotoğraf başarıyla gönderildi.")
        else:
            logger.error(f"Telegram: Gönderim başarısız! {r.text}")
    except Exception as e:
        logger.error(f"Telegram: Bağlantı hatası! {e}")

# ===========================================================
# BÖLGE SEÇİMİ (ROI)
# ===========================================================

def select_zones_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return []
    window_name = "BOLGE SECIMI (Cizdikten sonra ENTER'a bas, bitince ESC'bas)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    zones = cv2.selectROIs(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return [(z[0], z[1], z[0]+z[2], z[1]+z[3]) for z in zones if z[2] > 0 and z[3] > 0]

# ===========================================================
# ANALİZ ÇEKİRDEĞİ
# ===========================================================

def apply_kvkk_blur(frame, face_coords):
    h, w, _ = frame.shape
    if len(face_coords) > 0:
        x_pts = [p[0] for p in face_coords]; y_pts = [p[1] for p in face_coords]
        x1, x2 = max(0, min(x_pts)), min(w, max(x_pts))
        y1, y2 = max(0, min(y_pts)), min(h, max(y_pts))
        yuz = frame[y1:y2, x1:x2]
        if yuz.size > 0: frame[y1:y2, x1:x2] = cv2.GaussianBlur(yuz, (99, 99), 30)

def compute_ear_and_blur(frame, face_landmarks_list, track_boxes, violation_manager, current_sec, kvkk_modu=False):
    h, w, _ = frame.shape
    for face_landmarks in face_landmarks_list:
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        if kvkk_modu: apply_kvkk_blur(frame, coords)
        ear = (eye_aspect_ratio([coords[i] for i in LEFT_EYE_IDX]) + eye_aspect_ratio([coords[i] for i in RIGHT_EYE_IDX])) / 2.0
        nx, ny = coords[NOSE_IDX]
        for tid, bbox in track_boxes.items():
            if bbox[0] <= nx <= bbox[2] and bbox[1] <= ny <= bbox[3]:
                violation_manager.update_eye_state(tid, ear, current_sec)
                break

def process_pose_data(frame, pose_results, track_boxes, violation_manager, current_sec):
    h, w, _ = frame.shape
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        nose = (int(lm[POSE_NOSE_IDX].x * w), int(lm[POSE_NOSE_IDX].y * h))
        l_sh = (int(lm[POSE_LEFT_SHOULDER_IDX].x * w), int(lm[POSE_LEFT_SHOULDER_IDX].y * h))
        r_sh = (int(lm[POSE_RIGHT_SHOULDER_IDX].x * w), int(lm[POSE_RIGHT_SHOULDER_IDX].y * h))
        ratio = calculate_head_drop_ratio(nose, l_sh, r_sh)
        for tid, bbox in track_boxes.items():
            if bbox[0] <= nose[0] <= bbox[2] and bbox[1] <= nose[1] <= bbox[3]:
                violation_manager.update_head_state(tid, ratio, current_sec)
                break

def analyze_video_with_model(video_path, model_key, monitoring_zones=None, kvkk_modu=False):
    cfg = MODEL_CONFIGS[model_key]
    model = YOLO(cfg["engine"]) if os.path.exists(cfg["engine"]) else YOLO(cfg["pt"])
    cap = cv2.VideoCapture(video_path)
    video_placeholder = st.empty()
    tracker = CentroidTracker(max_distance=IOU_TRACK_THRESH)
    violation_manager = ViolationManager(
        still_threshold=STILLNESS_SECONDS, eye_threshold=EYE_CLOSED_SECONDS,
        movement_threshold=MOVEMENT_PIXEL_THRESHOLD, ear_threshold=EAR_THRESHOLD
    )
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    fps_video = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0; total_time_processing = 0.0; processed_frames = 0
    progress_bar = st.progress(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            current_sec = frame_index / fps_video if fps_video > 0 else 0.0
            frame_index += 1; start_time = time.time()

            yolo_results = model(frame, conf=CONF_THRESH, classes=[0], device=DEVICE, verbose=False)
            detections = []
            if len(yolo_results) > 0:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if monitoring_zones:
                        if not any(is_point_in_rect(((x1+x2)//2, (y1+y2)//2), z) for z in monitoring_zones): continue
                    detections.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            track_boxes = tracker.update(detections)
            violation_manager.update_tracks(track_boxes, current_sec)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(frame_rgb)
            if face_res.multi_face_landmarks:
                compute_ear_and_blur(frame, face_res.multi_face_landmarks, track_boxes, violation_manager, current_sec, kvkk_modu)
            pose_res = pose_detector.process(frame_rgb)
            process_pose_data(frame, pose_res, track_boxes, violation_manager, current_sec)

            is_global_crit, raw_reasons, per_person = violation_manager.compute_violations(current_sec)
            
            display_status = ""; status_color = (0, 255, 0)
            for tid, p_state in per_person.items():
                if p_state["deep_sleep_active"]:
                    display_status = "KESIN UYUYOR!"; status_color = (0, 0, 255)
                    if not p_state["telegram_sent"]:
                        telegram_foto_gonder(frame.copy(), "🔴 DİKKAT PATRON: Güvenlik Görevlisi Uyuyor! (60 sn. İhlal)")
                        violation_manager.person_states[tid].telegram_sent = True
                    break
                elif "Mesgul" in str(raw_reasons):
                    display_status = "MESGUL (Dikkat Dagildi)"; status_color = (0, 255, 255)
                elif raw_reasons:
                    display_status = " / ".join(raw_reasons); status_color = (0, 255, 255)

            if display_status:
                cv2.putText(frame, f"DURUM: {display_status}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                if status_color == (0, 0, 255): alarm_tetikle("acil")
                else: alarm_tetikle("uyari")
            else: pygame.mixer.stop()

            if monitoring_zones:
                for z in monitoring_zones: cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (0, 255, 255), 2)

            latency = (time.time() - start_time) * 1000
            fps_curr = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            cv2.putText(frame, f"FPS: {fps_curr:.1f} | {latency:.1f}ms", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            processed_frames += 1; total_time_processing += (time.time() - start_time)
            if total_frames > 0: progress_bar.progress(min(1.0, frame_index / total_frames))
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        violation_manager.finalize(current_sec)
    finally:
        cap.release(); face_mesh.close(); progress_bar.empty()
    return violation_manager.episodes, processed_frames / total_time_processing if total_time_processing > 0 else 0.0

# ===========================================================
# RAPORLAMA & DASHBOARD
# ===========================================================

def save_report_csv(model_results):
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    fpath = os.path.join(OUTPUT_REPORT_DIR, f"Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_data = []
    for mk, data in model_results.items():
        for ep in data["episodes"]:
            df_data.append({"model": MODEL_CONFIGS[mk]['label'], "ihlal_turu": ep.violation_type, "baslangic": ep.start_sec, "bitis": ep.end_sec, "sure": ep.duration})
    if df_data: pd.DataFrame(df_data).to_csv(fpath, index=False, sep=';', encoding='utf-8-sig')
    return fpath

def render_dashboard():
    st.subheader("📊 Günlük Özet & Analitik Dashboard")
    report_files = glob.glob(os.path.join(OUTPUT_REPORT_DIR, "Rapor_*.csv"))
    if not report_files:
        st.info("Henüz analiz verisi bulunamadı.")
        return

    all_dfs = []
    for f in report_files:
        try:
            tdf = pd.read_csv(f, sep=';', encoding='utf-8-sig')
            if 'ihlal' in tdf.columns: tdf = tdf.rename(columns={'ihlal': 'ihlal_turu'})
            fname = os.path.basename(f)
            parts = fname.split("_")
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                tdf['tarih'] = pd.to_datetime(date_str, format='%Y%m%d')
                tdf['saat'] = int(time_str[:2])
            all_dfs.append(tdf)
        except: continue
    
    if not all_dfs: return
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Metrikler
    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam İhlal Sayısı", len(df))
    m2.metric("Ortalama İhlal Süresi", f"{df['sure'].mean():.2f} sn")
    if not df.empty and 'saat' in df.columns:
        peak_hour = df['saat'].mode()[0]
        m3.metric("En Yoğun Saat Dilimi", f"{peak_hour}:00 - {peak_hour+1}:00")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("🕒 **Saatlik İhlal Dağılımı**")
        hourly_df = df.groupby('saat').size().reset_index(name='sayi')
        st.plotly_chart(px.bar(hourly_df, x='saat', y='sayi', color_discrete_sequence=['#ff4b4b']), use_container_width=True)

    with c2:
        st.write("📈 **Günlük İhlal Trendi**")
        trend_df = df.groupby('tarih').size().reset_index(name='sayi')
        st.plotly_chart(px.line(trend_df, x='tarih', y='sayi', markers=True), use_container_width=True)

    st.markdown("---")
    st.write("🤖 **Model Karşılaştırması (İhlal Türüne Göre)**")
    model_df = df.groupby(['model', 'ihlal_turu']).size().reset_index(name='sayi')
    st.plotly_chart(px.bar(model_df, x='model', y='sayi', color='ihlal_turu', barmode='group'), use_container_width=True)

# ===========================================================
# MAIN & ALARM
# ===========================================================

son_alarm_zamani = 0
def alarm_tetikle(tur):
    global son_alarm_zamani
    su_an = time.time()
    if ayarlar['alarm_settings']['enabled'] and (su_an - son_alarm_zamani > ayarlar['alarm_settings']['cooldown_seconds']):
        if tur == "acil" and acil_sesi: acil_sesi.play()
        elif uyari_sesi: uyari_sesi.play()
        son_alarm_zamani = su_an

def main():
    st.set_page_config(page_title="GuardWatch AI", layout="wide", page_icon="🌙")
    st.title("🌙 GuardWatch AI: Otonom Güvenlik Takip Sistemi")
    tab1, tab2 = st.tabs(["🎥 Video Analizi", "📊 Analitik Dashboard"])
    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("🛠️ Ayarlar")
            ayarlar['alarm_settings']['enabled'] = st.toggle("🚨 Alarm Sesleri", value=ayarlar['alarm_settings']['enabled'])
            use_roi = st.toggle("🎯 Bölge İzleme (ROI) Aktif", value=False)
            kvkk_modu = st.checkbox("🔒 KVKK Gizlilik Modu", value=True)
            selected_models = st.multiselect("Modeller", options=list(MODEL_CONFIGS.keys()), default=[list(MODEL_CONFIGS.keys())[0]], format_func=lambda k: MODEL_CONFIGS[k]["label"])
            uploaded_file = st.file_uploader("📂 Video Dosyası", type=["mp4"])
            start_btn = st.button("🚀 Analizi Başlat")
        with col2:
            if uploaded_file and start_btn:
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                m_zones = select_zones_from_video(temp_path) if use_roi else None
                results = {}
                for m_key in selected_models:
                    episodes, fps = analyze_video_with_model(temp_path, m_key, monitoring_zones=m_zones, kvkk_modu=kvkk_modu)
                    results[m_key] = {"episodes": episodes, "fps": fps}
                save_report_csv(results)
                st.success("Analiz tamamlandı.")
            else: st.info("Video yükleyerek analizi başlatın.")
    with tab2: render_dashboard()

if __name__ == "__main__": main()
