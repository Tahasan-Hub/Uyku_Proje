import os
import time
import math
import cv2
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from ultralytics import YOLO
import mediapipe as mp
import pygame

from utils.core_logic import (

    CentroidTracker, PersonState, ViolationManager as BaseViolationManager,

    eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX, is_point_in_rect

)



# ... (Previous constants)



def select_monitoring_zones(cap):

    """Kullanıcının ekran üzerinde birden fazla ROI çizmesini sağlar."""

    logger.info("Bölge seçimi ekranı açıldı. 'r' tuşuyla yeni bir bölge çizin, 'enter' ile bitirin.")

    ret, frame = cap.read()

    if not ret: return []

    

    # OpenCV'nin multi ROI seçicisi

    zones = cv2.selectROIs("Bolge Secimi (Enter ile bitir, ESC ile iptal)", frame, fromCenter=False, showCrosshair=True)

    cv2.destroyWindow("Bolge Secimi (Enter ile bitir, ESC ile iptal)")

    

    # cv2.selectROIs [x, y, w, h] formatında döner, biz [x1, y1, x2, y2] yapalım

    formatted_zones = []

    for i, z in enumerate(zones):

        x, y, w, h = z

        formatted_zones.append((x, y, x+w, y+h))

        logger.info(f"Bolge {i+1} tanimlandi: {z}")

    return formatted_zones



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

        

        # Görselleştirme: Bölgeleri çiz

        for i, zone in enumerate(monitoring_zones):

            cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), 2)

            cv2.putText(frame, f"Bolge {i+1}", (zone[0], zone[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


        
        # [INFO] Kisi tespit edildi
        for tid in track_boxes:
            if tid not in tracked_persons:
                logger.info(f"Kisi tespit edildi: ID={tid}")
                tracked_persons.add(tid)
        
        violation_manager.update_tracks(track_boxes, time.time())
        face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face_results.multi_face_landmarks:
            compute_ear_for_faces(frame, face_results.multi_face_landmarks, track_boxes, violation_manager)

        current_time_val = time.time()
        global_violation, reasons, per_person_timers = violation_manager.compute_violations(current_time_val)

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

        if global_violation:
            play_alert_sound(reasons)
            if violation_manager.should_save_frame():
                save_violation_frame(frame, reasons)
            cv2.putText(frame, "UYKU IHLALI!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            pygame.mixer.stop()

        # FPS & UI
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt > 0: fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 else 1.0 / dt
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Yapay Zeka Uyku Takibi", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("m"):
            global is_muted
            is_muted = not is_muted
            logger.info(f"Ses: {'KAPALI' if is_muted else 'ACIK'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
