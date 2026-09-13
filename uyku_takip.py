import locale
# KRİTİK: Türkçe locale'de (tr_TR) C kütüphanesinin tolower('I') -> 'ı' (noktasız i)
# davranışı, MediaPipe'ın C++ grafik adı üretimini bozuyor ve "TAG:index:name is invalid"
# hatasına yol açıyor. MediaPipe import edilmeden ÖNCE karakter sınıflandırmasını (LC_CTYPE)
# nötr 'C' locale'ine alıyoruz. Diğer locale ayarları (tarih/sayı biçimi) Türkçe kalır.
locale.setlocale(locale.LC_CTYPE, "C")

import os
import time
import cv2
import json
import logging
import requests
import numpy as np
from datetime import datetime

from ultralytics import YOLO
import mediapipe as mp
from utils.core_logic import (
    CentroidTracker, ViolationManager as BaseViolationManager,
    eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX, is_point_in_rect,
)
from utils.display import kisi_bbox_ciz
from utils.audio import AlarmPlayer
from utils.face_match import yuz_takip_eslestir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================
# MEDIAPIPE (Tasks API) ADAPTÖRÜ
# ===========================================================
# NOT: mediapipe >= 0.10.30 ile eski "mp.solutions.face_mesh" API'si kaldırıldı.
# Aşağıdaki adaptör yeni "Tasks" API'sini, kodun beklediği eski
# .process() / .multi_face_landmarks arayüzüyle birebir aynı şekilde sarar.
# Landmark indeksleri (478 yüz) eski API ile aynı olduğu için core_logic sabitleri değişmeden çalışır.
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from types import SimpleNamespace

MODEL_DIZINI = os.path.join(SCRIPT_DIR, "models")
YUZ_MODELI = os.path.join(MODEL_DIZINI, "face_landmarker.task")


class YuzAgiAdaptoru:
    """Eski mp.solutions.face_mesh.FaceMesh arayüzünü Tasks API üzerinde taklit eder."""

    def __init__(self, refine_landmarks=True, max_yuz=1):
        # cv2.selectROIs gibi GUI çağrıları LC_CTYPE'ı sistem (tr_TR) locale'ine geri
        # çevirebiliyor; grafik kurulmadan HEMEN ÖNCE nötr 'C'ye sabitliyoruz.
        locale.setlocale(locale.LC_CTYPE, "C")
        secenekler = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=YUZ_MODELI),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_yuz,
        )
        self._dedektor = mp_vision.FaceLandmarker.create_from_options(secenekler)

    def process(self, rgb_kare):
        mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_kare))
        sonuc = self._dedektor.detect(mp_goruntu)
        yuzler = [SimpleNamespace(landmark=lm) for lm in sonuc.face_landmarks] if sonuc.face_landmarks else None
        return SimpleNamespace(multi_face_landmarks=yuzler)

    def close(self):
        self._dedektor.close()


# Eski "mp_face_mesh.FaceMesh(refine_landmarks=True)" çağrısı bozulmadan çalışsın diye köprü.
class _YuzAgiFabrikasi:
    FaceMesh = YuzAgiAdaptoru


mp_face_mesh = _YuzAgiFabrikasi()


# ===========================================================
# YAPILANDIRMA (config.json)
# ===========================================================
with open(os.path.join(SCRIPT_DIR, "config.json"), "r", encoding="utf-8") as _f:
    ayarlar = json.load(_f)

CONF_THRESH = ayarlar.get("conf_threshold", 0.4)
IOU_TRACK_THRESH = ayarlar.get("iou_track_threshold", 100)
STILLNESS_SECONDS = ayarlar.get("stillness_seconds", 2.0)
EYE_CLOSED_SECONDS = ayarlar.get("eye_closed_seconds", 2.0)
MOVEMENT_PIXEL_THRESHOLD = ayarlar.get("movement_pixel_threshold", 20.0)
EAR_THRESHOLD = ayarlar.get("ear_threshold", 0.21)
SAVE_COOLDOWN_SECONDS = ayarlar.get("save_cooldown_seconds", 1.0)
OUTPUT_DIR = ayarlar.get("output_dir", "ihlal_kayitlari")
LOG_DIR = ayarlar.get("log_dir", "logs")

# CUDA (NVIDIA veya AMD ROCm) yoksa otomatik CPU'ya düş — "device=0" çökmesin.
try:
    import torch
    DEVICE = ayarlar.get("device", 0) if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ===========================================================
# LOGGING
# ===========================================================
os.makedirs(LOG_DIR, exist_ok=True)
_log_dosya = os.path.join(LOG_DIR, f"Log_{datetime.now().strftime('%Y-%m-%d')}.log")
logger = logging.getLogger("UykuTakipWebcam")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(_log_dosya, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]    %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_fh)

# ===========================================================
# SES (ALARM) — utils/audio.py (mutlak dosya yolu + Linux mixer)
# ===========================================================
alarm = AlarmPlayer(ayarlar, SCRIPT_DIR, logger)


# ===========================================================
# MODEL YÜKLEME (TRT engine varsa onu, yoksa .pt)
# ===========================================================

def load_or_build_trt_model():
    """Seçili modeli yükler: geçerli bir TensorRT engine varsa onu, değilse .pt ağırlığını kullanır."""
    secili = ayarlar.get("selected_model", "yolo11n")
    yapi = ayarlar["model_configs"][secili]
    engine_yolu, pt_yolu = yapi.get("engine"), yapi.get("pt")
    if engine_yolu and os.path.exists(engine_yolu):
        try:
            logger.info(f"TensorRT engine yukleniyor: {engine_yolu}")
            return YOLO(engine_yolu)
        except Exception as e:
            logger.warning(f"Engine yuklenemedi ({e}); .pt modeline geciliyor.")
    logger.info(f"PyTorch modeli yukleniyor: {pt_yolu}")
    return YOLO(pt_yolu)


# ===========================================================
# WEBCAM İHLAL YÖNETİCİSİ + EAR YARDIMCISI
# ===========================================================

class WebcamViolationManager(BaseViolationManager):
    """Canlı webcam için temel ViolationManager'a kare-kaydetme akış kısıtı (throttle) ekler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._son_kayit_zamani = 0.0

    def should_save_frame(self) -> bool:
        su_an = time.time()
        if su_an - self._son_kayit_zamani >= SAVE_COOLDOWN_SECONDS:
            self._son_kayit_zamani = su_an
            return True
        return False


def _landmarklardan_ear(landmarks, genislik: int, yukseklik: int) -> float:
    koordinatlar = [(int(lm.x * genislik), int(lm.y * yukseklik)) for lm in landmarks]
    return (
        eye_aspect_ratio([koordinatlar[i] for i in LEFT_EYE_IDX])
        + eye_aspect_ratio([koordinatlar[i] for i in RIGHT_EYE_IDX])
    ) / 2.0


def guncelle_goz_durumu(frame, track_boxes, face_mesh, violation_manager, current_time: float):
    """Tam kare + kişi kutusu kırpımı ile EAR; kapalı gözde yüz kaybını azaltır."""
    h, w, _ = frame.shape
    guncellenen_tidler = set()

    face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            koordinatlar = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            ear = _landmarklardan_ear(face_landmarks.landmark, w, h)
            tid = yuz_takip_eslestir(koordinatlar[NOSE_IDX], track_boxes)
            if tid is not None:
                violation_manager.update_eye_state(tid, ear, current_time)
                guncellenen_tidler.add(tid)

    for tid, (x1, y1, x2, y2) in track_boxes.items():
        if tid in guncellenen_tidler:
            continue
        bh, bw = max(1, y2 - y1), max(1, x2 - x1)
        kx1, ky1 = max(0, x1), max(0, y1)
        kx2, ky2 = min(w, x2), min(h, y1 + int(bh * 0.65))
        kirpim = frame[ky1:ky2, kx1:kx2]
        if kirpim.size == 0:
            continue
        kh, kw = kirpim.shape[:2]
        kirpim_sonuc = face_mesh.process(cv2.cvtColor(kirpim, cv2.COLOR_BGR2RGB))
        if kirpim_sonuc.multi_face_landmarks:
            ear = _landmarklardan_ear(kirpim_sonuc.multi_face_landmarks[0].landmark, kw, kh)
            violation_manager.update_eye_state(tid, ear, current_time)


def save_violation_frame(frame, reasons):
    """İhlal anının görüntüsünü diske kaydeder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    etiket = "_".join(str(r).replace(" ", "-").replace("!", "") for r in reasons) or "ihlal"
    dosya = os.path.join(OUTPUT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{etiket}.jpg")
    cv2.imwrite(dosya, frame)
    logger.info(f"Ihlal karesi kaydedildi: {dosya}")


def select_monitoring_zones(cap):

    """Kullanıcının ekran üzerinde birden fazla ROI çizmesini sağlar."""

    logger.info("Bölge seçimi: fare ile kutu çiz + ENTER/SPACE onayla, bitirmek için ESC.")

    ret, frame = cap.read()

    if not ret: return []

    

    # OpenCV'nin multi ROI seçicisi

    _pencere = "BOLGE SEC: kutu ciz + ENTER onayla | BITIRMEK icin ESC"

    zones = cv2.selectROIs(_pencere, frame, fromCenter=False, showCrosshair=True)

    cv2.destroyWindow(_pencere)

    cv2.waitKey(1)  # destroyWindow'un pencereyi gercekten kapatmasi icin GUI olayini isle

    

    # cv2.selectROIs [x, y, w, h] formatında döner, biz [x1, y1, x2, y2] yapalım

    formatted_zones = []

    for i, z in enumerate(zones):

        x, y, w, h = z

        formatted_zones.append((x, y, x+w, y+h))

        logger.info(f"Bolge {i+1} tanimlandi: {z}")

    return formatted_zones



def telegram_foto_gonder(frame, message, config, logger):
    """Telegram üzerinden fotoğraf ve mesaj gönderir."""
    t_cfg = config.get("telegram_settings", {})
    
    # Önce .env dosyasından oku, orada yoksa config.json'a bak
    token = os.getenv("TELEGRAM_BOT_TOKEN") or t_cfg.get("token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or t_cfg.get("chat_id")

    # Eğer ikisinde de yoksa veya içi boşsa bildirim göndermeyi iptal et
    if not token or not chat_id or "BURAYA" in str(token):
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    _, buffer = cv2.imencode(".jpg", frame)
    payload = {"chat_id": chat_id, "caption": message}
    files = {"photo": ("alarm.jpg", buffer.tobytes(), "image/jpeg")}
    
    try:
        r = requests.post(url, data=payload, files=files, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram: Bildirim başarıyla gönderildi.")
        else:
            logger.error(f"Telegram: Gönderim başarısız! {r.text}")
    except Exception as e:
        logger.error(f"Telegram: Bağlantı hatası! {e}")


def main():

    model = load_or_build_trt_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        logger.error("Kamera açılamadı.")

        return



    # GÖREV 5: ÇOKLU BÖLGE SEÇİMİ

    monitoring_zones = select_monitoring_zones(cap)

    if not monitoring_zones:

        logger.warning("Hic bolge secilmedi. Tum ekran izlenecek.")



    tracker = CentroidTracker(max_distance=IOU_TRACK_THRESH)

    violation_manager = WebcamViolationManager(

        still_threshold=STILLNESS_SECONDS,

        eye_threshold=EYE_CLOSED_SECONDS,

        movement_threshold=MOVEMENT_PIXEL_THRESHOLD,

        ear_threshold=EAR_THRESHOLD

    )

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)



    tracked_persons = set()

    warning_logged_still = set()

    warning_logged_eye = set()

    violation_logged_still = set()

    violation_logged_eye = set()



    last_time = time.time()

    fps = 0.0



    while True:

        ret, frame = cap.read()

        if not ret: break

        h, w, _ = frame.shape

        yolo_results = model(frame, conf=CONF_THRESH, classes=[0], device=DEVICE, verbose=False)

        

        raw_detections = [list(map(int, box.xyxy[0].tolist())) for box in yolo_results[0].boxes] if yolo_results[0].boxes is not None else []

        

        # GÖREV 5: BÖLGE FİLTRELEME

        filtered_detections = []

        if monitoring_zones:

            for det in raw_detections:

                cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2

                if any(is_point_in_rect((cx, cy), zone) for zone in monitoring_zones):

                    filtered_detections.append(det)

        else:

            filtered_detections = raw_detections



        track_boxes = tracker.update(filtered_detections)

        

        # [INFO] Kisi tespit edildi
        for tid in track_boxes:
            if tid not in tracked_persons:
                logger.info(f"Kisi tespit edildi: ID={tid}")
                tracked_persons.add(tid)
        
        current_time_val = time.time()
        violation_manager.update_tracks(track_boxes, current_time_val)
        guncelle_goz_durumu(frame, track_boxes, face_mesh, violation_manager, current_time_val)

        _, reasons, per_person_timers = violation_manager.compute_violations(current_time_val)

        # TELEGRAM MANTIĞI
        if "TELEGRAM_BILDIRIMI_GEREKLI" in reasons:
            reasons.remove("TELEGRAM_BILDIRIMI_GEREKLI")
            for tid, state in violation_manager.person_states.items():
                if not state.telegram_sent:
                    # Hangi ihlal 30s'yi geçtiyse ona göre mesaj yazalım
                    durum_mesaji = ""
                    if state.eye_violation_active and (current_time_val - state.eye_closed_start_time >= 30.0):
                        durum_mesaji = f"🔴 KRITIK: ID={tid} 30 saniyedir UYUYOR!"
                    elif state.still_violation_active and (current_time_val - state.still_start_time >= 30.0):
                        durum_mesaji = f"🟡 UYARI: ID={tid} 30 saniyedir HAREKETSIZ!"
                    
                    if durum_mesaji:
                        telegram_foto_gonder(frame.copy(), durum_mesaji, ayarlar, logger)
                        state.telegram_sent = True

        # REHBERE UYGUN SMART LOGGING
        for tid, state in violation_manager.person_states.items():
            # Hareketsizlik Süreci
            if state.still_start_time is not None:
                if tid not in warning_logged_still:
                    logger.warning(f"ID={tid} hareketsizlik suresi basladi")
                    warning_logged_still.add(tid)
                
                elapsed = current_time_val - state.still_start_time
                if elapsed >= STILLNESS_SECONDS and tid not in violation_logged_still:
                    logger.critical(f"ID={tid} HAREKETSIZLIK IHLALI ({STILLNESS_SECONDS} sn)")
                    violation_logged_still.add(tid)
            else:
                warning_logged_still.discard(tid)
                violation_logged_still.discard(tid)

            # Göz Kapalılığı Süreci
            if state.eye_closed_start_time is not None:
                if tid not in warning_logged_eye:
                    logger.warning(f"ID={tid} goz kapaliligi suresi basladi")
                    warning_logged_eye.add(tid)
                
                elapsed = current_time_val - state.eye_closed_start_time
                if elapsed >= EYE_CLOSED_SECONDS and tid not in violation_logged_eye:
                    logger.critical(f"ID={tid} GOZ KAPALI IHLALI ({EYE_CLOSED_SECONDS} sn)")
                    violation_logged_eye.add(tid)
            else:
                warning_logged_eye.discard(tid)
                violation_logged_eye.discard(tid)

        kisi_bbox_ciz(
            frame, track_boxes, violation_manager, per_person_timers,
            STILLNESS_SECONDS, EYE_CLOSED_SECONDS, EAR_THRESHOLD,
        )

        if reasons:
            alarm.guncelle(reasons)
            if violation_manager.should_save_frame():
                save_violation_frame(frame, reasons)
        else:
            alarm.guncelle([])

        # FPS
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt > 0: fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 else 1.0 / dt
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Yapay Zeka Uyku Takibi", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("m"):
            alarm.muted = not alarm.muted
            logger.info(f"Ses: {'KAPALI' if alarm.muted else 'ACIK'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
