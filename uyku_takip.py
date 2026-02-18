import os
import time
import math
import cv2
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from ultralytics import YOLO
import mediapipe as mp

from utils.core_logic import (
    CentroidTracker, PersonState, ViolationManager as BaseViolationManager,
    eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX
)


# ===========================================================
# MODEL & GENEL AYARLAR
# ===========================================================
# ... (sabitler aynı kalıyor) ...
PT_MODEL_PATH = "yolo11n.pt"          # YOLOv11n PyTorch ağırlık dosyası
ENGINE_MODEL_PATH = "yolo11n.engine"  # TensorRT engine dosyası
OUTPUT_DIR = "ihlal_kayitlari"        # İhlal kayıtlarının kaydedileceği klasör

DEVICE = 0  # 0: ilk GPU (TensorRT engine de buna göre derlenecek)
CONF_THRESH = 0.4  # Kişi tespiti için güven eşiği
IOU_TRACK_THRESH = 100  # Takip için maksimum merkez mesafe (piksel)

STILLNESS_SECONDS = 10.0      # Hareketsizlik ihlali için süre (sn)
EYE_CLOSED_SECONDS = 10.0     # Göz kapalı ihlali için süre (sn)
MOVEMENT_PIXEL_THRESHOLD = 20.0  # Hareketsizlik için merkez hareket eşiği (piksel)
EAR_THRESHOLD = 0.21          # Göz kapalılığı için EAR eşiği (kişiye göre ayarlanabilir)

SAVE_COOLDOWN_SECONDS = 1.0   # Aynı anda çok fazla kayıt almamak için minimum aralık (sn)


# ===========================================================
# ÖZEL İHLAL YÖNETİCİSİ (Webcam için Kaydetme Özellikli)
# ===========================================================

class WebcamViolationManager(BaseViolationManager):
    """
    Core ViolationManager'a ek olarak kare kaydetme cooldown kontrolü ekler.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_save_time: float = 0.0

    def should_save_frame(self) -> bool:
        current_time = time.time()
        if current_time - self.last_save_time >= SAVE_COOLDOWN_SECONDS:
            self.last_save_time = current_time
            return True
        return False


# ===========================================================
# EAR (Eye Aspect Ratio) HESABI - MediaPipe Face Mesh
# ===========================================================

# Resmi MediaPipe paketinde FaceMesh'e bu şekilde erişilir
mp_face_mesh = mp.solutions.face_mesh

def compute_ear_for_faces(
    frame: np.ndarray,
    face_landmarks_list,
    track_boxes: Dict[int, Tuple[int, int, int, int]],
    violation_manager: WebcamViolationManager,
):
    """
    - MediaPipe FaceMesh çıktısını kullanarak her yüz için EAR hesaplar.
    - Burun noktasının bulunduğu bounding box'a bakarak ilgili track_id'yi bulur.
    - Bulunan kişi için EAR değerini ViolationManager'a gönderir.
    """
    h, w, _ = frame.shape
    current_time = time.time()

    for face_landmarks in face_landmarks_list:
        coords: List[Tuple[int, int]] = []
        for lm in face_landmarks.landmark:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            coords.append((x_px, y_px))

        left_eye = [coords[i] for i in LEFT_EYE_IDX]
        right_eye = [coords[i] for i in RIGHT_EYE_IDX]

        ear = eye_aspect_ratio(left_eye + right_eye) if len(left_eye+right_eye) == 12 else 0.0
        # Not: core logic'teki eye_aspect_ratio 6 nokta bekliyor. 
        # Burada her gözü ayrı hesaplayıp ortalamasını alalım.
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
            violation_manager.update_eye_state(assigned_track_id, ear, current_time)


# ===========================================================
# YOLO + TENSORRT MODEL YÜKLEME / OLUŞTURMA
# ===========================================================

def load_or_build_trt_model() -> YOLO:
    """
    - Eğer mevcutsa TensorRT engine modelini (`.engine`) yükler.
    - Yoksa .pt modelini yükleyip TensorRT FP16 engine olarak export eder.
    - Sonrasında engine modelini kullanarak YOLO nesnesini döndürür.
    """
    # Mevcut engine dosyasını kontrol et
    if os.path.exists(ENGINE_MODEL_PATH):
        print(f"[INFO] Mevcut TensorRT engine bulundu: {ENGINE_MODEL_PATH}")
        model = YOLO(ENGINE_MODEL_PATH)
        return model

    # Engine yoksa, önce .pt dosyasını kontrol et
    if not os.path.exists(PT_MODEL_PATH):
        # Bu çağrı, ağırlıkları ultralytics'ten indirecektir
        print(f"[INFO] {PT_MODEL_PATH} bulunamadı, Ultralytics üzerinden indirilecek.")
        model = YOLO("yolo11n.pt")
    else:
        print(f"[INFO] PyTorch model yüklenecek: {PT_MODEL_PATH}")
        model = YOLO(PT_MODEL_PATH)

    print("[INFO] TensorRT FP16 engine üretiliyor (ilk seferde biraz sürebilir)...")
    # Ultralytics export fonksiyonu TensorRT engine oluşturur
    model.export(
        format="engine",
        half=True,   # FP16 hassasiyet
        device=DEVICE
    )

    # Varsayılan olarak `yolo11n.engine` oluşmasını bekliyoruz
    if not os.path.exists(ENGINE_MODEL_PATH):
        raise FileNotFoundError(
            f"{ENGINE_MODEL_PATH} oluşturulamadı. Export işleminde bir hata olmuş olabilir."
        )

    print(f"[INFO] TensorRT engine başarıyla oluşturuldu: {ENGINE_MODEL_PATH}")
    model = YOLO(ENGINE_MODEL_PATH)
    return model


# ===========================================================
# İHLAL KAYDI ALMA (GÖRÜNTÜ KAYDETME)
# ===========================================================

def save_violation_frame(frame: np.ndarray, reasons: List[str]):
    """
    İhlal anındaki kareyi, tarih/saat ve ihlal nedeni ile birlikte
    `ihlal_kayitlari` klasörüne JPEG olarak kaydeder.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    reason_text = " & ".join(reasons)
    filename = f"ihlal_{timestamp_str}_{reason_text.replace(' ', '_')}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Kaydedilecek kare üzerine metin yaz
    annotated = frame.copy()
    info_text = f"Tarih/Saat: {now.strftime('%Y-%m-%d %H:%M:%S')} - Neden: {reason_text}"

    cv2.putText(
        annotated,
        info_text,
        (10, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(filepath, annotated)
    print(f"[KAYIT] İhlal görüntüsü kaydedildi: {filepath}")


# ===========================================================
# ANA ÇALIŞMA DÖNGÜSÜ
# ===========================================================

def main():
    """
    Kameradan gelen görüntü üzerinde:
    - YOLO + TensorRT ile kişi tespiti
    - Basit centroid tracker ile ID atama
    - MediaPipe FaceMesh ile EAR (göz açıklık oranı) hesabı
    - 10 sn hareketsizlik veya göz kapalılığı durumunda ihlal tespiti ve kayıt
    - Ekranda FPS, EAR, geri sayım sayaçları ve ihlal uyarılarının gösterimi
    işlemlerini yapar.
    """
    # Modeli yükle veya ilk sefer için TensorRT engine oluştur
    model = load_or_build_trt_model()

    # Video kaynağını aç (0: varsayılan kamera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı.")
        return

    tracker = CentroidTracker(max_distance=IOU_TRACK_THRESH)
    violation_manager = WebcamViolationManager(
        still_threshold=STILLNESS_SECONDS,
        eye_threshold=EYE_CLOSED_SECONDS,
        movement_threshold=MOVEMENT_PIXEL_THRESHOLD,
        ear_threshold=EAR_THRESHOLD
    )

    # MediaPipe Face Mesh başlat
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fps = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kare okunamadı, döngü sonlandırılıyor.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # -----------------------------------
        # 1) YOLO ile kişi tespiti (TensorRT engine ile)
        # -----------------------------------
        yolo_results = model(
            frame,
            conf=CONF_THRESH,
            classes=[0],   # 0: person (COCO)
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

        # -----------------------------------
        # 2) Takip (ID atama)
        # -----------------------------------
        track_boxes = tracker.update(detections)
        violation_manager.update_tracks(track_boxes)

        # -----------------------------------
        # 3) MediaPipe FaceMesh ile EAR hesabı
        # -----------------------------------
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            compute_ear_for_faces(
                frame,
                face_results.multi_face_landmarks,
                track_boxes,
                violation_manager,
            )

        # -----------------------------------
        # 4) İhlal durumlarını hesapla
        # -----------------------------------
        global_violation, reasons, per_person_timers = violation_manager.compute_violations()

        # -----------------------------------
        # 5) Çizimler: bounding box, ID, sayaçlar, EAR
        # -----------------------------------
        for tid, bbox in track_boxes.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            person_info = per_person_timers.get(
                tid,
                {"still": STILLNESS_SECONDS, "eye": EYE_CLOSED_SECONDS, "ear": 0.0},
            )
            still_remaining = person_info["still"]
            eye_remaining = person_info["eye"]
            ear_val = person_info["ear"]

            # ID yazısı
            cv2.putText(
                frame,
                f"ID: {tid}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Hareketsizlik geri sayım
            cv2.putText(
                frame,
                f"Hareketsiz: {still_remaining:4.1f}s",
                (x1, y2 + 20 if y2 + 20 < h else y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Göz kapalılık geri sayım + EAR
            cv2.putText(
                frame,
                f"Goz Kapali: {eye_remaining:4.1f}s  EAR:{ear_val:.3f}",
                (x1, y2 + 40 if y2 + 40 < h else y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # -----------------------------------
        # 6) Global ihlal uyarısı ve kayıt
        # -----------------------------------
        if global_violation and reasons:
            warning_text = "UYKU IHLALI: " + " / ".join(reasons)
            cv2.putText(
                frame,
                warning_text,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            # İhlal anında kareyi kaydet (cooldown ile)
            if violation_manager.should_save_frame():
                save_violation_frame(frame, reasons)

        # -----------------------------------
        # 7) FPS hesabı ve ekranda gösterimi
        # -----------------------------------
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt > 0:
            fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 else 1.0 / dt

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # -----------------------------------
        # 8) Pencereyi göster ve klavye kontrolü
        # -----------------------------------
        cv2.imshow("Yapay Zeka Uyku ve Guvenlik Takip Sistemi", frame)

        # 'q' tuşu ile çıkış
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Kaynakları düzgün bir şekilde serbest bırak
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
