import os
import time
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import mediapipe as mp

from utils.core_logic import (
    CentroidTracker, ViolationManager, ViolationEpisode,
    eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX
)


# ===========================================================
# GENEL SABİTLER VE AYARLAR
# ===========================================================
# ... (sabitler aynı kalıyor) ...
MODEL_CONFIGS = {
    "yolo11n": {
        "pt": "yolo11n.pt",
        "engine": "yolo11n.engine",
        "label": "YOLO11n (nano)"
    },
    "yolo11s": {
        "pt": "yolo11s.pt",
        "engine": "yolo11s.engine",
        "label": "YOLO11s (small)"
    }
}

# İhlal ve takip ayarları
DEVICE = 0                     # GPU id
CONF_THRESH = 0.4              # YOLO güven eşiği
IOU_TRACK_THRESH = 100         # Centroid tracker max mesafe (piksel)
STILLNESS_SECONDS = 10.0       # Hareketsizlik ihlali eşiği (sn)
EYE_CLOSED_SECONDS = 10.0      # Göz kapalılığı ihlali eşiği (sn)
MOVEMENT_PIXEL_THRESHOLD = 20  # Hareketsizlik için merkez hareket eşiği (piksel)
EAR_THRESHOLD = 0.21           # Göz kapalı için EAR eşiği
OUTPUT_REPORT_DIR = "raporlar" # Raporların kaydedileceği klasör


# ===========================================================
# EAR / MEDIAPIPE FACE MESH
# ===========================================================

mp_face_mesh = mp.solutions.face_mesh

def compute_ear_for_faces(
    frame,
    face_landmarks_list,
    track_boxes: Dict[int, Tuple[int, int, int, int]],
    violation_manager: ViolationManager,
    current_sec: float,
):
    """
    - Her yüz için sol ve sağ göz EAR değerini hesaplar.
    - Burun noktasını kullanarak yüzü ilgili vücut kutusuna (track_id) bağlar.
    - EAR değerini ViolationManager'a iletir.
    """
    h, w, _ = frame.shape

    for face_landmarks in face_landmarks_list:
        coords = []
        for lm in face_landmarks.landmark:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            coords.append((x_px, y_px))

        left_eye = [coords[i] for i in LEFT_EYE_IDX]
        right_eye = [coords[i] for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        nose_x, nose_y = coords[NOSE_IDX]
        assigned_track_id = None
        for tid, bbox in track_boxes.items():
            x1, y1, x2, y2 = bbox
            if x1 <= nose_x <= x2 and y1 <= nose_y <= y2:
                assigned_track_id = tid
                break

        if assigned_track_id is not None:
            violation_manager.update_eye_state(assigned_track_id, ear, current_sec)


# ===========================================================
# YOLO + TENSORRT YÜKLEME/OLUŞTURMA
# ===========================================================

def load_or_build_trt_model(model_key: str) -> YOLO:
# ... (devamı aynı) ...
    """
    Belirtilen model için:
    - .engine varsa onu yükler.
    - Yoksa .pt'den TensorRT FP16 engine üretir, sonra onu yükler.
    """
    cfg = MODEL_CONFIGS[model_key]
    pt_path = cfg["pt"]
    engine_path = cfg["engine"]

    if os.path.exists(engine_path):
        st.info(f"{cfg['label']} için mevcut TensorRT engine bulundu: {engine_path}")
        return YOLO(engine_path)

    # Engine yoksa, önce PT modelini yükle (gerekirse indirir)
    if not os.path.exists(pt_path):
        st.warning(f"{pt_path} bulunamadı, Ultralytics üzerinden indirilecek.")
        model = YOLO(pt_path)  # isimle çağırmak indirir
    else:
        model = YOLO(pt_path)

    st.info(f"{cfg['label']} için TensorRT FP16 engine üretiliyor (ilk seferde sürebilir)...")
    model.export(format="engine", half=True, device=DEVICE)

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"{engine_path} oluşturulamadı.")

    st.success(f"{cfg['label']} TensorRT engine oluşturuldu: {engine_path}")
    return YOLO(engine_path)


# ===========================================================
# VİDEOYU BİR MODELLE ANALİZ ETME (ANA İŞLEMCİ FONKSİYON)
# ===========================================================

def analyze_video_with_model(video_path: str, model_key: str) -> Tuple[List[ViolationEpisode], float]:
    """
    Verilen video dosyasını tek bir YOLO modeli ile analiz eder.
    - Hareketsizlik ve göz kapalılık ihlallerini zaman damgalı olarak takip eder.
    - Biten ihlal epizodlarını (start, end, duration) döndürür.
    - Ortalama FPS (işleme hızı) döndürür.
    """
    model = load_or_build_trt_model(model_key)
    cfg = MODEL_CONFIGS[model_key]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı.")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = CentroidTracker(max_distance=IOU_TRACK_THRESH)
    violation_manager = ViolationManager(
        still_threshold=STILLNESS_SECONDS,
        eye_threshold=EYE_CLOSED_SECONDS,
        movement_threshold=MOVEMENT_PIXEL_THRESHOLD,
        ear_threshold=EAR_THRESHOLD
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # FPS ölçümü için
    total_time_processing = 0.0
    processed_frames = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_index = 0
    last_sec = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Video FPS'ine göre bu frame'in saniyesini hesapla
        current_sec = frame_index / fps_video if fps_video > 0 else 0.0
        last_sec = current_sec
        frame_index += 1

        start_time = time.time()

        h, w = frame.shape[:2]

        # 1) YOLO ile kişi tespiti (sadece "person" sınıfı)
        yolo_results = model(
            frame,
            conf=CONF_THRESH,
            classes=[0],  # person
            device=DEVICE,
            verbose=False
        )

        detections: List[Tuple[int, int, int, int]] = []
        if len(yolo_results) > 0:
            result = yolo_results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(max(0, x1))
                    y1 = int(max(0, y1))
                    x2 = int(min(w - 1, x2))
                    y2 = int(min(h - 1, y2))
                    detections.append((x1, y1, x2, y2))

        # 2) Centroid tracker ile ID takibi
        track_boxes = tracker.update(detections)
        violation_manager.update_tracks(track_boxes, current_sec)

        # 3) MediaPipe FaceMesh + EAR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            compute_ear_for_faces(
                frame,
                face_results.multi_face_landmarks,
                track_boxes,
                violation_manager,
                current_sec,
            )

        # 4) Global ihlal durumlarını güncelle
        violation_manager.compute_violations(current_sec)

        # FPS ölçümü
        end_time = time.time()
        total_time_processing += (end_time - start_time)
        processed_frames += 1

        # Streamlit progress bar güncelle
        if total_frames > 0:
            progress = min(1.0, frame_index / total_frames)
        else:
            progress = 0.0
        progress_bar.progress(progress)
        status_text.text(f"{cfg['label']} - İşlenen kare: {frame_index}/{total_frames} (t ~ {current_sec:.1f} sn)")

    # Video bittiğinde açık epizodları kapat
    violation_manager.finalize(last_sec)

    cap.release()
    face_mesh.close()

    avg_fps = processed_frames / total_time_processing if total_time_processing > 0 else 0.0
    progress_bar.empty()
    status_text.text(f"{cfg['label']} analizi tamamlandı. Ortalama FPS: {avg_fps:.2f}")

    return violation_manager.episodes, avg_fps


# ===========================================================
# RAPOR OLUŞTURMA
# ===========================================================

def save_report_csv(
    report_name: str,
    model_results: Dict[str, Dict[str, object]],
) -> str:
    """
    Verilen model sonuçlarıyla (ihlaller + fps) CSV raporu oluşturur.

    model_results:
      {
        "yolo11n": {
            "episodes": [ViolationEpisode, ...],
            "fps": 12.3
        },
        "yolo11s": {
            "episodes": [...],
            "fps": 18.7
        }
      }
    """
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{report_name}_{timestamp_str}.csv"
    filepath = os.path.join(OUTPUT_REPORT_DIR, filename)

    # Türkçe Windows/Excel'de genelde CSV ayıracı ";" olduğu için
    # burada da noktalı virgül kullanıyoruz. Sayıları da virgüllü
    # ondalık formatta yazarak Excel'de daha okunaklı hale getiriyoruz.
    sep = ";"

    lines = []
    # Başlık satırı
    lines.append(
        sep.join(
            [
                "model",
                "ihlal_turu",
                "baslangic_saniyesi",
                "bitis_saniyesi",
                "toplam_sure_saniye",
            ]
        )
    )

    # Her model için ihlalleri yaz
    for model_key, data in model_results.items():
        label = MODEL_CONFIGS[model_key]["label"]
        episodes: List[ViolationEpisode] = data.get("episodes", [])
        for ep in episodes:
            start_str = f"{ep.start_sec:.2f}".replace(".", ",")
            end_str = f"{ep.end_sec:.2f}".replace(".", ",")
            dur_str = f"{ep.duration:.2f}".replace(".", ",")
            lines.append(
                sep.join(
                    [
                        label,
                        ep.violation_type,
                        start_str,
                        end_str,
                        dur_str,
                    ]
                )
            )

    # Sonuna FPS özeti ekleyelim (ayrı blok)
    lines.append("")
    lines.append(sep.join(["model", "ortalama_fps"]))
    for model_key, data in model_results.items():
        label = MODEL_CONFIGS[model_key]["label"]
        fps = data.get("fps", 0.0)
        fps_str = f"{fps:.2f}".replace(".", ",")
        lines.append(sep.join([label, fps_str]))

    # Excel'in UTF-8'i doğru tanıyabilmesi için BOM ekleyen utf-8-sig kullanıyoruz.
    with open(filepath, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return filepath


# ===========================================================
# STREAMLIT ARAYÜZÜ
# ===========================================================

def main():
    """
    Streamlit tabanlı görsel arayüz:
    - Kullanıcıdan .mp4 video alır.
    - Hangi modeli/leri kullanacağını seçtirir.
    - Analiz başlatıldığında ilerleme çubuğu gösterir.
    - Analiz sonunda ihlal epizodlarını tablo ve CSV rapor olarak sunar.
    """
    st.set_page_config(page_title="Model Karşılaştırma ve Raporlama Sistemi", layout="wide")
    st.title("📊 Model Karşılaştırma ve Raporlama Sistemi")
    st.markdown("**YOLO11n vs YOLO11s - Uyku ve Güvenlik İhlal Analizi (TensorRT FP16)**")

    # Video yükleme
    uploaded_file = st.file_uploader("Analiz edilecek .mp4 videoyu seçin", type=["mp4"])

    # Model seçimi
    model_options = ["yolo11n", "yolo11s"]
    selected_models = st.multiselect(
        "Hangi modellerle analiz yapılsın?",
        options=model_options,
        default=model_options,  # varsayılan: ikisi de
        format_func=lambda k: MODEL_CONFIGS[k]["label"],
    )

    if not uploaded_file:
        st.info("Lütfen önce bir .mp4 video dosyası yükleyin.")
        return

    # Yüklenen dosyayı geçici bir yere kaydedelim
    temp_video_path = os.path.join("temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Analizi Başlat"):
        if not selected_models:
            st.warning("En az bir model seçmelisiniz.")
            return

        st.write("---")
        st.subheader("🔍 Analiz Başlıyor")

        model_results: Dict[str, Dict[str, object]] = {}

        # Her seçili model için ayrı ayrı analiz
        for m_key in selected_models:
            st.markdown(f"### {MODEL_CONFIGS[m_key]['label']} Analizi")
            episodes, avg_fps = analyze_video_with_model(temp_video_path, m_key)
            model_results[m_key] = {
                "episodes": episodes,
                "fps": avg_fps,
            }

            # Bu model için ihlalleri tablo olarak göster
            if episodes:
                # Streamlit tabloda tam olarak 2 ondalık hane göstermek için
                # değerleri string formatına çeviriyoruz.
                data = {
                    "İhlal Türü": [ep.violation_type for ep in episodes],
                    "Başlangıç (sn)": [f"{ep.start_sec:.2f}" for ep in episodes],
                    "Bitiş (sn)": [f"{ep.end_sec:.2f}" for ep in episodes],
                    "Süre (sn)": [f"{ep.duration:.2f}" for ep in episodes],
                }
                st.table(data)
            else:
                st.info("Bu model için ihlal tespit edilmedi.")

            st.write(f"**Ortalama FPS:** {avg_fps:.2f}")
            st.write("---")

        # Her iki model için performans karşılaştırması
        st.subheader("⚖️ Model Performans Karşılaştırması (Ort. FPS)")
        # Ortalama FPS'leri tabloda da tam 2 ondalık hane ile göstermek için
        # string formatına çeviriyoruz.
        perf_data = {
            "Model": [MODEL_CONFIGS[m]["label"] for m in model_results.keys()],
            "Ortalama FPS": [f"{model_results[m]['fps']:.2f}" for m in model_results.keys()],
        }
        st.table(perf_data)

        # Rapor dosyasını oluştur ve indirme linki ver
        st.subheader("📁 Rapor Oluşturma")
        report_path = save_report_csv("model_karsilastirma_raporu", model_results)
        st.success(f"Rapor oluşturuldu: {report_path}")

        with open(report_path, "rb") as f:
            st.download_button(
                label="Raporu İndir (.csv)",
                data=f,
                file_name=os.path.basename(report_path),
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

