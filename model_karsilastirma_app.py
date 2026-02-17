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


# ===========================================================
# GENEL SABİTLER VE AYARLAR
# ===========================================================

# Model isimleri ve dosya yolları (n ve s)
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
# BASİT ID TAKİPÇİ (CENTROID TRACKER)
# ===========================================================

@dataclass
class Track:
    """Her bir kişi (takip edilen nesne) için ID ve bbox bilgisi tutar."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_update_time: float = field(default_factory=time.time)


class CentroidTracker:
    """
    Çok basit centroid tabanlı takip:
    - Her karede gelen bounding box'ları, bir önceki karedeki track'lerle
      merkez mesafesine göre eşleştirir.
    - Her kişiye benzersiz bir track_id atar.
    """

    def __init__(self, max_distance: float = IOU_TRACK_THRESH, max_lost_time: float = 1.0):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_lost_time = max_lost_time

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Verilen bounding box'ın merkez noktasını hesaplar."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Yeni tespitlere göre track'leri günceller ve her birine ID atar.

        :param detections: Her biri (x1, y1, x2, y2) bounding box listesi
        :return: track_id -> bbox sözlüğü
        """
        current_time = time.time()
        assigned_tracks: Dict[int, Tuple[int, int, int, int]] = {}

        # Tespit yoksa sadece zaman aşımı kontrolü yap
        if not detections:
            self._cleanup(current_time)
            return {tid: t.bbox for tid, t in self.tracks.items()}

        track_ids = list(self.tracks.keys())
        track_centers = [self._center_of_box(self.tracks[tid].bbox) for tid in track_ids]
        used_detections = set()

        # Her yeni tespit için en yakın track'i bul
        for det_idx, det_bbox in enumerate(detections):
            det_center = self._center_of_box(det_bbox)
            best_track_id = None
            best_dist = float("inf")

            for tid, t_center in zip(track_ids, track_centers):
                if tid in assigned_tracks:
                    continue
                dist = math.dist(det_center, t_center)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                self.tracks[best_track_id].bbox = det_bbox
                self.tracks[best_track_id].last_update_time = current_time
                assigned_tracks[best_track_id] = det_bbox
                used_detections.add(det_idx)

        # Eşleşmeyen her tespit için yeni track oluştur
        for det_idx, det_bbox in enumerate(detections):
            if det_idx in used_detections:
                continue
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = Track(track_id=new_id, bbox=det_bbox, last_update_time=current_time)
            assigned_tracks[new_id] = det_bbox

        # Eski, kaybolmuş track'leri temizle
        self._cleanup(current_time)

        return {tid: self.tracks[tid].bbox for tid in self.tracks.keys()}

    def _cleanup(self, current_time: float):
        """Uzun süre güncellenmeyen (kayıp) track'leri siler."""
        to_delete = []
        for tid, t in self.tracks.items():
            if current_time - t.last_update_time > self.max_lost_time:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]


# ===========================================================
# KİŞİ DURUM TAKİBİ VE İHLAL EPİZODLARI
# ===========================================================

@dataclass
class PersonState:
    """
    Her kişi için:
    - referans merkez (hareketsizlik için),
    - hareketsizlik başlangıç zamanı,
    - göz kapalılık başlangıç zamanı,
    - son EAR değeri,
    gibi bilgileri tutar.
    """
    track_id: int
    ref_centroid: Tuple[float, float]
    still_start_time: Optional[float] = None
    eye_closed_start_time: Optional[float] = None
    last_ear: float = 0.0


@dataclass
class ViolationEpisode:
    """
    Bir ihlal epizodunu temsil eder:
    - tür (Hareketsizlik / Göz Kapalı),
    - başlangıç saniyesi,
    - bitiş saniyesi,
    - toplam süre (sn).
    """
    violation_type: str
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class ViolationManager:
    """
    Tüm kişiler için ihlalleri takip eder ve
    global ihlal epizodlarını (başlangıç/bitiş saniyeleri) kaydeder.
    """

    def __init__(self):
        self.person_states: Dict[int, PersonState] = {}

        # Global ihlal durumları (her tür için ayrı)
        self.global_still_active = False
        self.global_eye_active = False
        self.global_still_start_sec: Optional[float] = None
        self.global_eye_start_sec: Optional[float] = None

        # Biten ihlal epizodları
        self.episodes: List[ViolationEpisode] = []

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update_tracks(self, track_boxes: Dict[int, Tuple[int, int, int, int]], current_sec: float):
        """
        Her karede, aktif track listesine göre:
        - yeni PersonState'ler oluşturur,
        - hareketsizlik sürelerini günceller,
        - kaybolan kişileri siler.
        """
        existing_ids = set(self.person_states.keys())
        current_ids = set(track_boxes.keys())

        # Yeni track'ler
        for tid in current_ids - existing_ids:
            centroid = self._center_of_box(track_boxes[tid])
            self.person_states[tid] = PersonState(track_id=tid, ref_centroid=centroid)

        # Hareketsizlik takibi
        for tid in current_ids:
            bbox = track_boxes[tid]
            centroid = self._center_of_box(bbox)
            state = self.person_states[tid]

            dist = math.dist(centroid, state.ref_centroid)
            if dist < MOVEMENT_PIXEL_THRESHOLD:
                if state.still_start_time is None:
                    state.still_start_time = current_sec
            else:
                state.ref_centroid = centroid
                state.still_start_time = None

        # Kaybolan track'leri sil
        for tid in existing_ids - current_ids:
            del self.person_states[tid]

    def update_eye_state(self, track_id: int, ear: float, current_sec: float):
        """
        MediaPipe FaceMesh'ten gelen EAR değerine göre
        ilgili kişide göz kapalılık süresini günceller.
        """
        if track_id not in self.person_states:
            return

        state = self.person_states[track_id]
        state.last_ear = ear

        if ear < EAR_THRESHOLD:
            if state.eye_closed_start_time is None:
                state.eye_closed_start_time = current_sec
        else:
            state.eye_closed_start_time = None

    def compute_global_violations(self, current_sec: float):
        """
        Tüm kişiler için:
        - Hareketsizlik ve göz kapalılık süresini kontrol eder,
        - global ihlal durumunu günceller,
        - ihlal epizodlarını (başlangıç/bitiş saniyeleriyle) kaydeder.
        """
        # En az bir kişide ihlal var mı?
        any_still_violation = False
        any_eye_violation = False

        for state in self.person_states.values():
            # Hareketsizlik kontrolü
            if state.still_start_time is not None:
                if current_sec - state.still_start_time >= STILLNESS_SECONDS:
                    any_still_violation = True

            # Göz kapalılık kontrolü
            if state.eye_closed_start_time is not None:
                if current_sec - state.eye_closed_start_time >= EYE_CLOSED_SECONDS:
                    any_eye_violation = True

        # --- Hareketsizlik epizod yönetimi ---
        if any_still_violation:
            if not self.global_still_active:
                # Yeni bir hareketsizlik epizodu başlıyor
                self.global_still_active = True
                self.global_still_start_sec = current_sec
        else:
            if self.global_still_active:
                # Hareketsizlik epizodu sona erdi → kaydet
                start = self.global_still_start_sec if self.global_still_start_sec is not None else current_sec
                self.episodes.append(ViolationEpisode("Hareketsizlik", start_sec=start, end_sec=current_sec))
                self.global_still_active = False
                self.global_still_start_sec = None

        # --- Göz kapalılık epizod yönetimi ---
        if any_eye_violation:
            if not self.global_eye_active:
                self.global_eye_active = True
                self.global_eye_start_sec = current_sec
        else:
            if self.global_eye_active:
                start = self.global_eye_start_sec if self.global_eye_start_sec is not None else current_sec
                self.episodes.append(ViolationEpisode("Goz Kapali", start_sec=start, end_sec=current_sec))
                self.global_eye_active = False
                self.global_eye_start_sec = None

    def finalize(self, last_sec: float):
        """
        Video bittiğinde, hala açık olan epizodlar varsa
        son saniyeyi bitiş olarak kabul edip kapatır.
        """
        if self.global_still_active and self.global_still_start_sec is not None:
            self.episodes.append(
                ViolationEpisode("Hareketsizlik", start_sec=self.global_still_start_sec, end_sec=last_sec)
            )
        if self.global_eye_active and self.global_eye_start_sec is not None:
                self.episodes.append(
                    ViolationEpisode("Goz Kapali", start_sec=self.global_eye_start_sec, end_sec=last_sec)
                )


# ===========================================================
# EAR / MEDIAPIPE FACE MESH
# ===========================================================

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_IDX = 1  # yüzü vücut kutusuna bağlamak için basit burun noktası


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
    """
    EAR hesabı:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Göz kapandıkça vertical mesafeler küçülür → EAR düşer.
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

    tracker = CentroidTracker()
    violation_manager = ViolationManager()

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
        violation_manager.compute_global_violations(current_sec)

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

