"""Pygame alarm: tek kanal, ihlal başına bir kez — göz→siren, hareket→beep."""
import os
from typing import Iterable, Optional

import pygame

_GOZ_ANAHTARLARI = ("goz", "uyku", "uyuyor", "kesin uyuyor", "kafa")


class AlarmPlayer:
    def __init__(self, ayarlar: dict, script_dir: str, logger=None):
        self.ayarlar = ayarlar
        self.logger = logger
        self._muted = False
        self._beep: Optional[pygame.mixer.Sound] = None
        self._siren: Optional[pygame.mixer.Sound] = None
        self._kanal: Optional[pygame.mixer.Channel] = None
        self._aktif_tip: Optional[str] = None  # "goz" | "hareket" | None

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        self._kanal = pygame.mixer.Channel(0)

        alarm = ayarlar.get("alarm_settings", {})
        beep_yolu = os.path.join(script_dir, alarm.get("uyari_sesi", "beep.wav"))
        siren_yolu = os.path.join(script_dir, alarm.get("acil_durum_sesi", "Siren.wav"))
        try:
            if os.path.isfile(beep_yolu):
                self._beep = pygame.mixer.Sound(beep_yolu)
            if os.path.isfile(siren_yolu):
                self._siren = pygame.mixer.Sound(siren_yolu)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ses yuklenemedi: {e}")

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool):
        self._muted = value
        if value:
            self.durdur()

    def _goz_ihlali_mi(self, reasons: Iterable[str]) -> bool:
        for r in reasons:
            rl = str(r).lower()
            if any(k in rl for k in _GOZ_ANAHTARLARI):
                return True
        return False

    def _tip_belirle(self, reasons: Iterable[str]) -> Optional[str]:
        nedenler = list(reasons)
        if not nedenler:
            return None
        return "goz" if self._goz_ihlali_mi(nedenler) else "hareket"

    def durdur(self):
        if self._kanal is not None:
            self._kanal.stop()
        self._aktif_tip = None

    def guncelle(self, reasons: Iterable[str]):
        """İhlal tipi değişince eski sesi durdur; aynı ihlal sürerken kanal boşalırsa tekrar çal."""
        if self._muted or not self.ayarlar.get("alarm_settings", {}).get("enabled", True):
            self.durdur()
            return

        yeni_tip = self._tip_belirle(reasons)
        if yeni_tip is None:
            self.durdur()
            return

        # Kanal çalıyorsa ve tip aynıysa bekle
        if self._kanal is not None and self._kanal.get_busy() and yeni_tip == self._aktif_tip:
            return

        # Tip değiştiyse veya kanal sustuysa (tekrar çalma)
        if yeni_tip != self._aktif_tip:
            self.durdur()

        ses = self._siren if yeni_tip == "goz" else self._beep
        if ses is not None and self._kanal is not None:
            self._kanal.play(ses)
            self._aktif_tip = yeni_tip
