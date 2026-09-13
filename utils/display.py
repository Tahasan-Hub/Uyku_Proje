"""Kişi bbox: yeşil=normal, sarı=hareketsiz, kırmızı=göz kapalı/uyuyor."""
from typing import Dict, Tuple

import cv2

from utils.core_logic import PersonState, ViolationManager


def _ust_etiket(state: PersonState) -> str:
    if state.eye_violation_active:
        return "UYUYOR"
    if state.still_violation_active:
        return "HAREKETSIZ"
    return "Aktif"


def _bbox_rengi(state: PersonState) -> Tuple[int, int, int]:
    if state.eye_violation_active:
        return (0, 0, 255)  # Kırmızı
    if state.still_violation_active:
        return (0, 255, 255)  # Sarı
    return (0, 255, 0)  # Yeşil


def kisi_bbox_ciz(
    frame,
    track_boxes: Dict[int, Tuple[int, int, int, int]],
    violation_manager: ViolationManager,
    per_person_timers: Dict[int, dict],
    still_threshold: float,
    eye_threshold: float,
    ear_threshold: float = 0.21,
):
    h = frame.shape[0]
    for tid, (x1, y1, x2, y2) in track_boxes.items():
        state = violation_manager.person_states.get(tid)
        info = per_person_timers.get(tid, {})
        if state is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            continue

        renk = _bbox_rengi(state)
        kalinlik = 3 if renk == (0, 0, 255) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, kalinlik)

        etiket = _ust_etiket(state)
        ust = f"{etiket} (ID:{tid})"
        cv2.putText(frame, ust, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, renk, 2, cv2.LINE_AA)

        satir_y = min(h - 5, y2 + 20)
        
        # Süre gösterimi
        if state.eye_closed_start_time is not None:
            eye_el = info.get("eye_elapsed", 0.0)
            cv2.putText(
                frame, f"Goz Kapali: {eye_el:.1f}s",
                (x1, satir_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )
        elif state.still_start_time is not None:
            still_el = info.get("still_elapsed", 0.0)
            cv2.putText(
                frame, f"Hareketsiz: {still_el:.1f}s",
                (x1, satir_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "Durum: AKTIF",
                (x1, satir_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
