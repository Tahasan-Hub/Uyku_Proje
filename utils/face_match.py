"""MediaPipe yüzünü YOLO kişi kutusuna bağlar."""
import math
from typing import Dict, Optional, Tuple


def yuz_takip_eslestir(
    burun: Tuple[int, int],
    takip_kutulari: Dict[int, Tuple[int, int, int, int]],
    margin: float = 0.3,
) -> Optional[int]:
    """
    Burnu en uygun track ID ile eşleştirir.
    Tek kişi varsa doğrudan o ID'ye bağlanır (webcam demosu için kritik).
    """
    if not takip_kutulari:
        return None

    if len(takip_kutulari) == 1:
        return next(iter(takip_kutulari))

    bx, by = burun
    en_iyi_id: Optional[int] = None
    en_iyi_mesafe = float("inf")

    for tid, (x1, y1, x2, y2) in takip_kutulari.items():
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        px1, py1 = x1 - w * margin, y1 - h * margin
        px2, py2 = x2 + w * margin, y2 + h * margin
        if px1 <= bx <= px2 and py1 <= by <= py2:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            mesafe = math.hypot(bx - cx, by - cy)
            if mesafe < en_iyi_mesafe:
                en_iyi_mesafe = mesafe
                en_iyi_id = tid

    if en_iyi_id is not None:
        return en_iyi_id

    # Son çare: en yakın kutu merkezi (üst yarı — yüz genelde üstte)
    for tid, (x1, y1, x2, y2) in takip_kutulari.items():
        cx = (x1 + x2) / 2.0
        cy = y1 + (y2 - y1) * 0.25
        mesafe = math.hypot(bx - cx, by - cy)
        w = max(1, x2 - x1)
        if mesafe < w and mesafe < en_iyi_mesafe:
            en_iyi_mesafe = mesafe
            en_iyi_id = tid

    return en_iyi_id
