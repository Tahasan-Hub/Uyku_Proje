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


# ===========================================================
# MODEL & GENEL AYARLAR
# ===========================================================

# Bu dosya: NVIDIA GPU + TensorRT ile optimize edilmiş
# basit bir "Yapay Zeka Uyku ve Güvenlik Takip Sistemi" örneğidir.
# - YOLOv11n (ultralytics) + TensorRT FP16 engine
# - Kişi tespiti + basit takip (ID)
# - 10 sn hareketsizlik ihlali
# - 10 sn göz kapalılığı (EAR) ihlali
# - İhlal anında ekran görüntüsü kaydı

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
# BASİT ID TAKİPÇİ (Centroid Tracker)
# ===========================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    last_update_time: float = field(default_factory=time.time)


class CentroidTracker:
    """
    Çok basit bir centroid tabanlı takip sınıfı.
    - Her karede gelen bounding box'ları, bir önceki karedeki track'lere
      merkez mesafesine göre eşler.
    - Temel senaryolarda yeterlidir, üretim için daha gelişmiş takipçiler kullanılabilir.
    """

    def __init__(self, max_distance: float = IOU_TRACK_THRESH, max_lost_time: float = 1.0):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_lost_time = max_lost_time

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        :param detections: Her biri (x1, y1, x2, y2) olan bounding box listesi
        :return: track_id -> bbox sözlüğü
        """
        current_time = time.time()
        assigned_tracks: Dict[int, Tuple[int, int, int, int]] = {}

        # Eğer hiç tespit yoksa, sadece zaman aşımına uğrayan track'leri temizle.
        if not detections:
            self._cleanup(current_time)
            return {tid: t.bbox for tid, t in self.tracks.items()}

        # Mevcut track'lerin merkezlerini hazırla
        track_ids = list(self.tracks.keys())
        track_centers = [self._center_of_box(self.tracks[tid].bbox) for tid in track_ids]

        used_detections = set()

        # Her yeni bounding box için en uygun (en yakın) track'i bul
        for det_idx, det_bbox in enumerate(detections):
            det_center = self._center_of_box(det_bbox)
            best_track_id = None
            best_dist = float("inf")

            for tid, t_center in zip(track_ids, track_centers):
                # Aynı karede bir track'e iki kez atama yapma
                if tid in assigned_tracks:
                    continue
                dist = math.dist(det_center, t_center)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                # Mevcut track'i güncelle
                self.tracks[best_track_id].bbox = det_bbox
                self.tracks[best_track_id].last_update_time = current_time
                assigned_tracks[best_track_id] = det_bbox
                used_detections.add(det_idx)

        # Eşleşmemiş tespitler için yeni track ID oluştur
        for det_idx, det_bbox in enumerate(detections):
            if det_idx in used_detections:
                continue
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = Track(track_id=new_id, bbox=det_bbox, last_update_time=current_time)
            assigned_tracks[new_id] = det_bbox

        # Zaman aşımına uğramış track'leri sil
        self._cleanup(current_time)

        # Güncel track'leri döndür
        return {tid: self.tracks[tid].bbox for tid in self.tracks.keys()}

    def _cleanup(self, current_time: float):
        """Uzun süre güncellenmeyen (kayıp) track'leri temizler."""
        to_delete = []
        for tid, t in self.tracks.items():
            if current_time - t.last_update_time > self.max_lost_time:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]


# ===========================================================
# KİŞİ DURUM TAKİBİ (Hareketsizlik + Göz Kapalı)
# ===========================================================

@dataclass
class PersonState:
    """
    Her bir kişi (track_id) için tutulacak durum bilgisi:
    - ref_centroid: Hareketsizlik karşılaştırması için referans merkez
    - still_start_time: Hareketsizlik süresinin başladığı zaman
    - eye_closed_start_time: Göz kapalılığı süresinin başladığı zaman
    - last_ear: Son ölçülen EAR değeri
    """

    track_id: int
    ref_centroid: Tuple[float, float]
    still_start_time: Optional[float] = None
    eye_closed_start_time: Optional[float] = None
    last_ear: float = 0.0
    still_violation_active: bool = False
    eye_violation_active: bool = False


class ViolationManager:
    """
    Her kişi için:
    - Hareketsizlik süresi
    - Göz kapalılığı süresi
    - EAR değeri
    takibini yapar ve global ihlal durumunu hesaplar.
    """

    def __init__(self):
        self.person_states: Dict[int, PersonState] = {}
        self.last_save_time: float = 0.0

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update_tracks(self, track_boxes: Dict[int, Tuple[int, int, int, int]]):
        """
        - Yeni track'ler için PersonState oluşturur.
        - Mevcut track'ler için merkez bilgisiyle hareketsizlik durumunu günceller.
        - Kaybolan track'leri siler.
        """
        existing_ids = set(self.person_states.keys())
        current_ids = set(track_boxes.keys())

        # Yeni gelen track'ler için başlangıç durumu oluştur
        for tid in current_ids - existing_ids:
            centroid = self._center_of_box(track_boxes[tid])
            self.person_states[tid] = PersonState(track_id=tid, ref_centroid=centroid)

        # Hareketsizlik için merkez hareketini takip et
        current_time = time.time()
        for tid in current_ids:
            bbox = track_boxes[tid]
            centroid = self._center_of_box(bbox)
            state = self.person_states[tid]

            dist = math.dist(centroid, state.ref_centroid)
            if dist < MOVEMENT_PIXEL_THRESHOLD:
                # Kişi yeterince sabit görünüyorsa süreyi başlat/koru
                if state.still_start_time is None:
                    state.still_start_time = current_time
            else:
                # Kişi hareket etti, referans noktayı ve süreyi sıfırla
                state.ref_centroid = centroid
                state.still_start_time = None
                state.still_violation_active = False

        # Artık görünmeyen track'leri temizle
        for tid in existing_ids - current_ids:
            del self.person_states[tid]

    def update_eye_state(self, track_id: int, ear: float):
        """
        Göz açıklık oranını (EAR) günceller ve göz kapalılık süresini takip eder.
        """
        current_time = time.time()
        if track_id not in self.person_states:
            return

        state = self.person_states[track_id]
        state.last_ear = ear

        if ear < EAR_THRESHOLD:
            # Gözler kapalı kabul ediliyor
            if state.eye_closed_start_time is None:
                state.eye_closed_start_time = current_time
        else:
            # Gözler tekrar açıldı, süreyi ve ihlal durumunu sıfırla
            state.eye_closed_start_time = None
            state.eye_violation_active = False

    def compute_violations(self) -> Tuple[bool, List[str], Dict[int, Dict[str, float]]]:
        """
        Tüm kişiler için ihlalleri hesaplar.

        :return:
            - global_violation: Herhangi bir ihlal var mı?
            - reasons: Tespit edilen ihlal nedenleri (metin listesi)
            - per_person_timers: {track_id: {"still": kalan_süre, "eye": kalan_süre, "ear": EAR}}
        """
        current_time = time.time()
        global_violation = False
        reasons = set()
        per_person_timers: Dict[int, Dict[str, float]] = {}

        for tid, state in self.person_states.items():
            still_remaining = STILLNESS_SECONDS
            eye_remaining = EYE_CLOSED_SECONDS

            # Hareketsizlik ihlali
            if state.still_start_time is not None:
                elapsed_still = current_time - state.still_start_time
                still_remaining = max(0.0, STILLNESS_SECONDS - elapsed_still)
                if elapsed_still >= STILLNESS_SECONDS:
                    global_violation = True
                    state.still_violation_active = True
                    reasons.add("Hareketsizlik")

                # Goz kapalilik ihlali
            if state.eye_closed_start_time is not None:
                elapsed_eye = current_time - state.eye_closed_start_time
                eye_remaining = max(0.0, EYE_CLOSED_SECONDS - elapsed_eye)
                if elapsed_eye >= EYE_CLOSED_SECONDS:
                    global_violation = True
                    state.eye_violation_active = True
                    reasons.add("Goz Kapali")

            per_person_timers[tid] = {
                "still": still_remaining,
                "eye": eye_remaining,
                "ear": state.last_ear,
            }

        return global_violation, sorted(list(reasons)), per_person_timers

    def should_save_frame(self) -> bool:
        """
        Çok sık kayıt alınmasını engellemek için basit bir cooldown kontrolü.
        """
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

# MediaPipe FaceMesh (468 nokta) için tipik göz landmark index'leri
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Yüz konumu için burun çevresinden bir nokta (yaklaşık merkez)
NOSE_IDX = 1


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """İki nokta arasındaki Öklid uzaklığını hesaplar."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio (EAR) hesabı:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    6 noktalı göz çevresi koordinatları kullanılır.
    """
    if len(eye_points) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def compute_ear_for_faces(
    frame: np.ndarray,
    face_landmarks_list,
    track_boxes: Dict[int, Tuple[int, int, int, int]],
    violation_manager: ViolationManager,
):
    """
    - MediaPipe FaceMesh çıktısını kullanarak her yüz için EAR hesaplar.
    - Burun noktasının bulunduğu bounding box'a bakarak ilgili track_id'yi bulur.
    - Bulunan kişi için EAR değerini ViolationManager'a gönderir.
    """
    h, w, _ = frame.shape

    for face_landmarks in face_landmarks_list:
        # Landmark'ları piksel koordinatına çevir
        coords: List[Tuple[int, int]] = []
        for lm in face_landmarks.landmark:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            coords.append((x_px, y_px))

        # Sol ve sağ göz noktalarını seç
        left_eye = [coords[i] for i in LEFT_EYE_IDX]
        right_eye = [coords[i] for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Burun noktası ile hangi kişiye ait olduğunu bul (basit kutu içi kontrol)
        nose_x, nose_y = coords[NOSE_IDX]
        assigned_track_id = None
        for tid, bbox in track_boxes.items():
            x1, y1, x2, y2 = bbox
            if x1 <= nose_x <= x2 and y1 <= nose_y <= y2:
                assigned_track_id = tid
                break

        if assigned_track_id is not None:
            violation_manager.update_eye_state(assigned_track_id, ear)


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

    tracker = CentroidTracker()
    violation_manager = ViolationManager()

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
