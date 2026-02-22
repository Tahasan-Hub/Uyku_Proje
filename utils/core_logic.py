import time
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

# ===========================================================
# MEDIAPIPE SABİTLERİ
# ===========================================================
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_IDX = 1  # Yüzü vücut kutusuna bağlamak için

# POSE LANDMARKS (Kafa Düşmesi için)
POSE_NOSE_IDX = 0
POSE_LEFT_SHOULDER_IDX = 11
POSE_RIGHT_SHOULDER_IDX = 12

# ===========================================================
# EAR (Eye Aspect Ratio) VE KAFA DÜŞMESİ HESABI
# ===========================================================

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """İki nokta arasındaki Öklid uzaklığını hesaplar."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
    """
    EAR hesabı:
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

def calculate_head_drop_ratio(nose: Tuple[float, float], l_shoulder: Tuple[float, float], r_shoulder: Tuple[float, float]) -> float:
    """
    Kafa Düşmesi Oranı: Burnun omuz hattına olan dikey mesafesinin, omuz genişliğine oranı.
    Bu oran kafanın öne düşmesiyle azalır.
    """
    shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    shoulder_width = euclidean_distance(l_shoulder, r_shoulder)
    
    if shoulder_width == 0:
        return 1.0
        
    vertical_dist = shoulder_center_y - nose[1]
    return vertical_dist / shoulder_width

# ===========================================================
# TAKİP YAPILARI (TRACKING)
# ===========================================================

@dataclass
class Track:
    """Her bir kişi (takip edilen nesne) için ID ve bbox bilgisi tutar."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_update_time: float = field(default_factory=time.time)

class CentroidTracker:
    """
    Basit centroid tabanlı takip:
    - Tespitleri merkez mesafelerine göre eşleştirir ve benzersiz ID'ler atar.
    """
    def __init__(self, max_distance: float = 100, max_lost_time: float = 1.0):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_lost_time = max_lost_time

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        current_time = time.time()
        assigned_tracks: Dict[int, Tuple[int, int, int, int]] = {}

        if not detections:
            self._cleanup(current_time)
            return {tid: t.bbox for tid, t in self.tracks.items()}

        track_ids = list(self.tracks.keys())
        track_centers = [self._center_of_box(self.tracks[tid].bbox) for tid in track_ids]
        used_detections = set()

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

        for det_idx, det_bbox in enumerate(detections):
            if det_idx in used_detections:
                continue
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = Track(track_id=new_id, bbox=det_bbox, last_update_time=current_time)
            assigned_tracks[new_id] = det_bbox

        self._cleanup(current_time)
        return {tid: self.tracks[tid].bbox for tid in self.tracks.keys()}

    def _cleanup(self, current_time: float):
        to_delete = [tid for tid, t in self.tracks.items() if current_time - t.last_update_time > self.max_lost_time]
        for tid in to_delete:
            del self.tracks[tid]

# ===========================================================
# İHLAL VE DURUM YÖNETİMİ
# ===========================================================

@dataclass
class PersonState:
    """Her kişi için durumsal bilgileri tutar."""
    track_id: int
    ref_centroid: Tuple[float, float]
    still_start_time: Optional[float] = None
    eye_closed_start_time: Optional[float] = None
    head_drop_start_time: Optional[float] = None
    last_ear: float = 0.0
    last_head_drop_ratio: float = 1.0
    still_violation_active: bool = False
    eye_violation_active: bool = False
    head_violation_active: bool = False
    busy_status_active: bool = False
    deep_sleep_active: bool = False
    telegram_sent: bool = False # Anti-Spam: Bu olay için Telegram mesajı atıldı mı?

@dataclass
class ViolationEpisode:
    """Bir ihlal epizodunu temsil eder."""
    violation_type: str
    start_sec: float
    end_sec: Optional[float] = None
    @property
    def duration(self) -> float:
        if self.end_sec is None: return 0.0
        return max(0.0, self.end_sec - self.start_sec)

class ViolationManager:
    """
    İhlal durumlarını hem gerçek zamanlı hem de bölümler halinde takip eder.
    """
    def __init__(self, still_threshold=10.0, eye_threshold=10.0, movement_threshold=20.0, ear_threshold=0.21, head_drop_threshold=0.15):
        self.person_states: Dict[int, PersonState] = {}
        self.episodes: List[ViolationEpisode] = []
        
        # Thresholds
        self.STILLNESS_SECONDS = still_threshold
        self.EYE_CLOSED_SECONDS = eye_threshold
        self.MOVEMENT_PIXEL_THRESHOLD = movement_threshold
        self.EAR_THRESHOLD = ear_threshold
        self.HEAD_DROP_THRESHOLD = head_drop_threshold
        
        # Temporal Analysis (Kademeli Zaman) Thresholds
        self.BUSY_THRESHOLD = 10.0 # 10 saniye boyunca "Meşgul"
        self.DEEP_SLEEP_THRESHOLD = 60.0 # 60 saniye boyunca "KESİN UYUYOR"

        # Global States (For Episode tracking)
        self.global_still_active = False
        self.global_eye_active = False
        self.global_head_active = False
        self.global_busy_active = False
        self.global_deep_sleep_active = False
        
        self.global_still_start_sec: Optional[float] = None
        self.global_eye_start_sec: Optional[float] = None
        self.global_head_start_sec: Optional[float] = None
        self.global_busy_start_sec: Optional[float] = None
        self.global_deep_sleep_start_sec: Optional[float] = None

    @staticmethod
    def _center_of_box(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update_tracks(self, track_boxes: Dict[int, Tuple[int, int, int, int]], current_time: float):
        existing_ids = set(self.person_states.keys())
        current_ids = set(track_boxes.keys())

        for tid in current_ids - existing_ids:
            centroid = self._center_of_box(track_boxes[tid])
            self.person_states[tid] = PersonState(track_id=tid, ref_centroid=centroid)

        for tid in current_ids:
            bbox = track_boxes[tid]
            centroid = self._center_of_box(bbox)
            state = self.person_states[tid]
            dist = math.dist(centroid, state.ref_centroid)

            if dist < self.MOVEMENT_PIXEL_THRESHOLD:
                if state.still_start_time is None:
                    state.still_start_time = current_time
            else:
                state.ref_centroid = centroid
                state.still_start_time = None
                state.still_violation_active = False
                state.busy_status_active = False
                state.deep_sleep_active = False
                state.telegram_sent = False # Uyandı veya hareket ettiyse sıfırla

        for tid in existing_ids - current_ids:
            del self.person_states[tid]

    def update_eye_state(self, track_id: int, ear: float, current_time: float):
        if track_id not in self.person_states:
            return
        state = self.person_states[track_id]
        state.last_ear = ear

        if ear < self.EAR_THRESHOLD:
            if state.eye_closed_start_time is None:
                state.eye_closed_start_time = current_time
        else:
            state.eye_closed_start_time = None
            state.eye_violation_active = False
            state.deep_sleep_active = False
            state.telegram_sent = False # Gözü açtıysa sıfırla

    def update_head_state(self, track_id: int, ratio: float, current_time: float):
        if track_id not in self.person_states:
            return
        state = self.person_states[track_id]
        state.last_head_drop_ratio = ratio

        if ratio < self.HEAD_DROP_THRESHOLD:
            if state.head_drop_start_time is None:
                state.head_drop_start_time = current_time
        else:
            state.head_drop_start_time = None
            state.head_violation_active = False
            state.deep_sleep_active = False
            state.telegram_sent = False # Kafayı kaldırdıysa sıfırla

    def compute_violations(self, current_time: float) -> Tuple[bool, List[str], Dict[int, Dict[str, float]]]:
        """Gerçek zamanlı uyarılar için durum hesaplar."""
        global_violation = False
        reasons = set()
        per_person_timers: Dict[int, Dict[str, float]] = {}

        any_still = False
        any_eye = False
        any_head = False
        any_busy = False
        any_deep_sleep = False

        for tid, state in self.person_states.items():
            still_rem = self.STILLNESS_SECONDS
            eye_rem = self.EYE_CLOSED_SECONDS
            head_rem = 2.0 # Kafa düşmesi için kısa süreli bekleme

            # --- STANDART IHLALLER ---
            if state.still_start_time is not None:
                elapsed = current_time - state.still_start_time
                still_rem = max(0.0, self.STILLNESS_SECONDS - elapsed)
                if elapsed >= self.STILLNESS_SECONDS:
                    state.still_violation_active = True
                    reasons.add("Hareketsizlik")
                    any_still = True

            if state.eye_closed_start_time is not None:
                elapsed = current_time - state.eye_closed_start_time
                eye_rem = max(0.0, self.EYE_CLOSED_SECONDS - elapsed)
                if elapsed >= self.EYE_CLOSED_SECONDS:
                    state.eye_violation_active = True
                    reasons.add("Goz Kapali")
                    any_eye = True

            if state.head_drop_start_time is not None:
                elapsed = current_time - state.head_drop_start_time
                head_rem = max(0.0, 2.0 - elapsed)
                if elapsed >= 2.0:
                    state.head_violation_active = True
                    reasons.add("Kafa Dusmesi")
                    any_head = True

            # --- GUARDWATCH AI: KESİN UYUYOR (Kafa Düşmüş + Göz Kapalı) ---
            if state.head_drop_start_time is not None and state.eye_closed_start_time is not None:
                # İkisi birden aktif. Başlangıç zamanı ikisinden sonuncusu (aralıksız ikisinin başladığı an)
                start_both = max(state.head_drop_start_time, state.eye_closed_start_time)
                elapsed_both = current_time - start_both
                
                if elapsed_both >= self.DEEP_SLEEP_THRESHOLD:
                    state.deep_sleep_active = True
                    reasons.add("KESİN UYUYOR!")
                    any_deep_sleep = True
                    global_violation = True # Kritik alarm

            # --- MEŞGUL DURUMU (Örn. sadece kafa düşmüş veya sadece göz kapalı + hareketsiz) ---
            elif (state.head_drop_start_time is not None or state.eye_closed_start_time is not None) and state.still_start_time is not None:
                elapsed_suspicious = current_time - state.still_start_time
                if elapsed_suspicious >= self.BUSY_THRESHOLD:
                    state.busy_status_active = True
                    reasons.add("Mesgul (Telefonda/Kitapta)")
                    any_busy = True

            per_person_timers[tid] = {
                "still": still_rem, 
                "eye": eye_rem, 
                "head": head_rem,
                "ear": state.last_ear,
                "head_ratio": state.last_head_drop_ratio,
                "deep_sleep_active": state.deep_sleep_active,
                "telegram_sent": state.telegram_sent
            }

        # Epizod (Bölüm) takibi
        self._update_global_episodes(any_still, any_eye, any_head, any_busy, any_deep_sleep, current_time)

        return global_violation, sorted(list(reasons)), per_person_timers

    def _update_global_episodes(self, any_still: bool, any_eye: bool, any_head: bool, any_busy: bool, any_deep_sleep: bool, current_time: float):
        # Hareketsizlik Epizodu
        if any_still:
            if not self.global_still_active:
                self.global_still_active = True
                self.global_still_start_sec = current_time
        else:
            if self.global_still_active:
                start = self.global_still_start_sec if self.global_still_start_sec is not None else current_time
                self.episodes.append(ViolationEpisode("Hareketsizlik", start, current_time))
                self.global_still_active = False
                self.global_still_start_sec = None

        # Göz Kapalı Epizodu
        if any_eye:
            if not self.global_eye_active:
                self.global_eye_active = True
                self.global_eye_start_sec = current_time
        else:
            if self.global_eye_active:
                start = self.global_eye_start_sec if self.global_eye_start_sec is not None else current_time
                self.episodes.append(ViolationEpisode("Goz Kapali", start, current_time))
                self.global_eye_active = False
                self.global_eye_start_sec = None

        # Kafa Düşmesi Epizodu
        if any_head:
            if not self.global_head_active:
                self.global_head_active = True
                self.global_head_start_sec = current_time
        else:
            if self.global_head_active:
                start = self.global_head_start_sec if self.global_head_start_sec is not None else current_time
                self.episodes.append(ViolationEpisode("Kafa Dusmesi", start, current_time))
                self.global_head_active = False
                self.global_head_start_sec = None

        # Meşgul Epizodu
        if any_busy:
            if not self.global_busy_active:
                self.global_busy_active = True
                self.global_busy_start_sec = current_time
        else:
            if self.global_busy_active:
                start = self.global_busy_start_sec if self.global_busy_start_sec is not None else current_time
                self.episodes.append(ViolationEpisode("Mesgul", start, current_time))
                self.global_busy_active = False
                self.global_busy_start_sec = None

        # KESİN UYUYOR Epizodu
        if any_deep_sleep:
            if not self.global_deep_sleep_active:
                self.global_deep_sleep_active = True
                self.global_deep_sleep_start_sec = current_time
        else:
            if self.global_deep_sleep_active:
                start = self.global_deep_sleep_start_sec if self.global_deep_sleep_start_sec is not None else current_time
                self.episodes.append(ViolationEpisode("KESIN UYUYOR!", start, current_time))
                self.global_deep_sleep_active = False
                self.global_deep_sleep_start_sec = None

    def finalize(self, last_time: float):
        """Biten video veya kapanan sistem için açık kalan epizodları kapatır."""
        if self.global_still_active and self.global_still_start_sec is not None:
            self.episodes.append(ViolationEpisode("Hareketsizlik", self.global_still_start_sec, last_time))
        if self.global_eye_active and self.global_eye_start_sec is not None:
            self.episodes.append(ViolationEpisode("Goz Kapali", self.global_eye_start_sec, last_time))
        if self.global_head_active and self.global_head_start_sec is not None:
            self.episodes.append(ViolationEpisode("Kafa Dusmesi", self.global_head_start_sec, last_time))
        if self.global_busy_active and self.global_busy_start_sec is not None:
            self.episodes.append(ViolationEpisode("Mesgul", self.global_busy_start_sec, last_time))
        if self.global_deep_sleep_active and self.global_deep_sleep_start_sec is not None:
            self.episodes.append(ViolationEpisode("KESIN UYUYOR!", self.global_deep_sleep_start_sec, last_time))

def is_point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    """Bir noktanın (x, y) verilen dikdörtgen (x1, y1, x2, y2) içinde olup olmadığını kontrol eder."""
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2
