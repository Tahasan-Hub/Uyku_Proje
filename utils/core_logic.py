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

# ===========================================================
# EAR (Eye Aspect Ratio) HESABI
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
    last_ear: float = 0.0
    still_violation_active: bool = False
    eye_violation_active: bool = False

@dataclass
class ViolationEpisode:
    """Bir ihlal epizodunu temsil eder."""
    violation_type: str
    start_sec: float
    end_sec: float
    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

class ViolationManager:
    """
    İhlal durumlarını hem gerçek zamanlı hem de bölümler halinde takip eder.
    """
    def __init__(self, still_threshold=10.0, eye_threshold=10.0, movement_threshold=20.0, ear_threshold=0.21):
        self.person_states: Dict[int, PersonState] = {}
        self.episodes: List[ViolationEpisode] = []
        
        # Thresholds
        self.STILLNESS_SECONDS = still_threshold
        self.EYE_CLOSED_SECONDS = eye_threshold
        self.MOVEMENT_PIXEL_THRESHOLD = movement_threshold
        self.EAR_THRESHOLD = ear_threshold

        # Global States (For Episode tracking)
        self.global_still_active = False
        self.global_eye_active = False
        self.global_still_start_sec: Optional[float] = None
        self.global_eye_start_sec: Optional[float] = None

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

    def compute_violations(self, current_time: float) -> Tuple[bool, List[str], Dict[int, Dict[str, float]]]:
        """Gerçek zamanlı uyarılar için durum hesaplar."""
        global_violation = False
        reasons = set()
        per_person_timers: Dict[int, Dict[str, float]] = {}

        any_still = False
        any_eye = False

        for tid, state in self.person_states.items():
            still_rem = self.STILLNESS_SECONDS
            eye_rem = self.EYE_CLOSED_SECONDS

            if state.still_start_time is not None:
                elapsed = current_time - state.still_start_time
                still_rem = max(0.0, self.STILLNESS_SECONDS - elapsed)
                if elapsed >= self.STILLNESS_SECONDS:
                    global_violation = True
                    state.still_violation_active = True
                    reasons.add("Hareketsizlik")
                    any_still = True

            if state.eye_closed_start_time is not None:
                elapsed = current_time - state.eye_closed_start_time
                eye_rem = max(0.0, self.EYE_CLOSED_SECONDS - elapsed)
                if elapsed >= self.EYE_CLOSED_SECONDS:
                    global_violation = True
                    state.eye_violation_active = True
                    reasons.add("Goz Kapali")
                    any_eye = True

            per_person_timers[tid] = {"still": still_rem, "eye": eye_rem, "ear": state.last_ear}

        # Epizod (Bölüm) takibini de burada güncelliyoruz (Hem canlı hem app için)
        self._update_global_episodes(any_still, any_eye, current_time)

        return global_violation, sorted(list(reasons)), per_person_timers

    def _update_global_episodes(self, any_still: bool, any_eye: bool, current_time: float):
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

    def finalize(self, last_time: float):
        """Biten video veya kapanan sistem için açık kalan epizodları kapatır."""
        if self.global_still_active and self.global_still_start_sec is not None:
            self.episodes.append(ViolationEpisode("Hareketsizlik", self.global_still_start_sec, last_time))
        if self.global_eye_active and self.global_eye_start_sec is not None:
            self.episodes.append(ViolationEpisode("Goz Kapali", self.global_eye_start_sec, last_time))
