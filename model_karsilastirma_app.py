import locale
# KRİTİK: Türkçe locale'de (tr_TR) C kütüphanesinin tolower('I') -> 'ı' (noktasız i)
# davranışı, MediaPipe'ın C++ grafik adı üretimini bozuyor ve "TAG:index:name is invalid"
# hatasına yol açıyor. MediaPipe import edilmeden ÖNCE karakter sınıflandırmasını (LC_CTYPE)
# nötr 'C' locale'ine alıyoruz. Diğer locale ayarları (tarih/sayı biçimi) Türkçe kalır.
locale.setlocale(locale.LC_CTYPE, "C")

import os
import time
import requests
import json
import logging
import glob
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv

# .env dosyasındaki gizli değişkenleri ortama yükler
load_dotenv()

from utils.core_logic import (
    CentroidTracker, ViolationManager,
    eye_aspect_ratio, calculate_head_drop_ratio, is_point_in_rect,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX,
    POSE_NOSE_IDX, POSE_LEFT_SHOULDER_IDX, POSE_RIGHT_SHOULDER_IDX,
)
from utils.display import kisi_bbox_ciz
from utils.audio import AlarmPlayer
from utils.face_match import yuz_takip_eslestir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================
# CONFIGURATION & LOGGING
# ===========================================================
try:
    with open(os.path.join(SCRIPT_DIR, "config.json"), "r", encoding="utf-8") as f:
        ayarlar = json.load(f)
except Exception as e:
    st.error(f"config.json okunamadı: {e}")
    st.stop()

LOG_DIZINI = ayarlar.get("log_dir", "logs")
os.makedirs(LOG_DIZINI, exist_ok=True)
log_dosya_adi = os.path.join(LOG_DIZINI, f"Log_{datetime.now().strftime('%Y-%m-%d')}.log")

gunlukcu = logging.getLogger("UykuTakipApp")
gunlukcu.setLevel(logging.INFO)
if not gunlukcu.handlers:
    file_handler = logging.FileHandler(log_dosya_adi, encoding='utf-8')
    file_format = logging.Formatter('%(asctime)s [%(levelname)s]    %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    gunlukcu.addHandler(file_handler)

# ===========================================================
# SABİTLER
# ===========================================================
MODEL_YAPILANDIRMALARI = ayarlar['model_configs']
CIHAZ = ayarlar.get("device", 0)
GUVEN_ESIGI = ayarlar.get("conf_threshold", 0.4)
IOU_TAKIP_ESIGI = ayarlar.get("iou_track_threshold", 100)
HAREKETSIZLIK_SANIYESI = ayarlar.get("stillness_seconds", 2.0)
GOZ_KAPALI_SANIYESI = ayarlar.get("eye_closed_seconds", 2.0)
HAREKET_PIKSEL_ESIGI = ayarlar.get("movement_pixel_threshold", 20.0)
EAR_ESIGI = ayarlar.get("ear_threshold", 0.21)
CIKTI_RAPOR_DIZINI = ayarlar.get("output_report_dir", "raporlar")

# NOT: mediapipe >= 0.10.30 ile eski "mp.solutions" (face_mesh / pose) API'si
# tamamen kaldırıldı. Aşağıdaki adaptörler yeni "Tasks" API'sini, kodun geri
# kalanının beklediği eski .process() / .multi_face_landmarks / .pose_landmarks
# arayüzüyle birebir aynı şekilde sarar. Landmark indeksleri (478 yüz, 33 poz)
# eski API ile aynı olduğu için core_logic sabitleri değişmeden çalışır.
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from types import SimpleNamespace

MODEL_DIZINI = os.path.join(SCRIPT_DIR, "models")
YUZ_MODELI = os.path.join(MODEL_DIZINI, "face_landmarker.task")
POZ_MODELI = os.path.join(MODEL_DIZINI, "pose_landmarker.task")


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


class PozAdaptoru:
    """Eski mp.solutions.pose.Pose arayüzünü Tasks API üzerinde taklit eder."""

    def __init__(self, max_poz=1):
        # cv2.selectROIs gibi GUI çağrıları LC_CTYPE'ı sistem (tr_TR) locale'ine geri
        # çevirebiliyor; grafik kurulmadan HEMEN ÖNCE nötr 'C'ye sabitliyoruz.
        locale.setlocale(locale.LC_CTYPE, "C")
        secenekler = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POZ_MODELI),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=max_poz,
        )
        self._dedektor = mp_vision.PoseLandmarker.create_from_options(secenekler)

    def process(self, rgb_kare):
        mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_kare))
        sonuc = self._dedektor.detect(mp_goruntu)
        poz = SimpleNamespace(landmark=sonuc.pose_landmarks[0]) if sonuc.pose_landmarks else None
        return SimpleNamespace(pose_landmarks=poz)

    def close(self):
        self._dedektor.close()


# Eski "mp_face_mesh.FaceMesh(refine_landmarks=True)" çağrısı bozulmadan çalışsın diye köprü.
class _YuzAgiFabrikasi:
    FaceMesh = YuzAgiAdaptoru


mp_face_mesh = _YuzAgiFabrikasi()
poz_dedektoru = PozAdaptoru()
alarm = AlarmPlayer(ayarlar, SCRIPT_DIR, gunlukcu)

# ===========================================================
# TELEGRAM & SNAPSHOT
# ===========================================================

def telegram_foto_gonder(cerceve, mesaj):
    """
        Kamera karesinde tespit edilen ihlali fotoğraf olarak Telegram kanalına/kullanıcısına yollar.
        Token ve Chat ID kodun içine gömülmez, güvenli bir şekilde .env dosyasından okunur
    """
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    Chat_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Güvenlik kontrolü: Eğer .env dosyası yoksa veya token girilmemişse sistemi çökertme, uyar ve çık
    if not TOKEN or not Chat_ID:
        gunlukcu.error("Telegram token veya chat_id bulunamadı! .env dosyanızı kontrol edin.")
        return

    # KVKK ve Veri Hazırlığı: OpenCV'nin BGR formatındaki görüntüsünü JPEG formatında bayt dizisine çeviriyoruz
    # Hafızadaki bir görüntüyü diske kaydetmeden doğrudan internete göndermek için imencodee kullanıyoruz (hızlı ve temiz) 
    _, ara_bellek = cv2.imencode(".jpg", cerceve)
    if not _:
        gunlukcu.error("Görüntü JPEG formatına dönüştürülemedi!")
        return

    # Telegram Bot API'sinin fotoğraf yükleme uç noktası
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    payload = {"chat_id": Chat_ID, "caption": mesaj}
    dosyalar = {"photo": ("guardwatch_alarm.jpg", ara_bellek.tobytes(), "image/jpeg")}

    # İsteği gönderiyoruz ve zaman aşımı (timeout) koyuyoruz ki internet kesilirse uygulama donmasın
    try:
        r = requests.post(url, data=payload, files=dosyalar, timeout=10)
        if r.status_code == 200:
            gunlukcu.info("Telegram: Fotoğraf başarıyla gönderildi.")
        else:
            gunlukcu.error(f"Telegram: Gönderim başarısız! {r.text}")
    except Exception as e:
        gunlukcu.error(f"Telegram: Bağlantı hatası! {e}")

# ===========================================================
# BÖLGE SEÇİMİ (ROI)
# ===========================================================

def videodan_bolgeleri_sec(video_yolu):
    yakalama = cv2.VideoCapture(video_yolu)
    durum, cerceve = yakalama.read()
    yakalama.release()
    if not durum: return []
    pencere_adi = "BOLGE SECIMI (Cizdikten sonra ENTER'a bas, bitince ESC'bas)"
    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(pencere_adi, cv2.WND_PROP_TOPMOST, 1)
    bolgeler = cv2.selectROIs(pencere_adi, cerceve, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return [(z[0], z[1], z[0]+z[2], z[1]+z[3]) for z in bolgeler if z[2] > 0 and z[3] > 0]

# ===========================================================
# ANALİZ ÇEKİRDEĞİ
# ===========================================================

def kvkk_bulaniklik_uygula(cerceve, yuz_koordinatlari):
    h, w, _ = cerceve.shape
    if len(yuz_koordinatlari) > 0:
        x_noktalari = [p[0] for p in yuz_koordinatlari]; y_noktalari = [p[1] for p in yuz_koordinatlari]
        x1, x2 = max(0, min(x_noktalari)), min(w, max(x_noktalari))
        y1, y2 = max(0, min(y_noktalari)), min(h, max(y_noktalari))
        yuz = cerceve[y1:y2, x1:x2]
        if yuz.size > 0: cerceve[y1:y2, x1:x2] = cv2.GaussianBlur(yuz, (99, 99), 30)

def ear_hesapla_ve_bulaniklastir(cerceve, yuz_isaret_listesi, takip_kutulari, ihlal_yoneticisi, su_anki_sn, kvkk_modu=False):
    h, w, _ = cerceve.shape
    for yuz_isaretleri in yuz_isaret_listesi:
        koordinatlar = [(int(lm.x * w), int(lm.y * h)) for lm in yuz_isaretleri.landmark]
        if kvkk_modu: kvkk_bulaniklik_uygula(cerceve, koordinatlar)
        ear = (eye_aspect_ratio([koordinatlar[i] for i in LEFT_EYE_IDX]) + eye_aspect_ratio([koordinatlar[i] for i in RIGHT_EYE_IDX])) / 2.0
        tid = yuz_takip_eslestir(koordinatlar[NOSE_IDX], takip_kutulari)
        if tid is not None:
            ihlal_yoneticisi.update_eye_state(tid, ear, su_anki_sn)

def poz_verisini_isle(cerceve, poz_sonuclari, takip_kutulari, ihlal_yoneticisi, su_anki_sn):
    h, w, _ = cerceve.shape
    if poz_sonuclari.pose_landmarks:
        lm = poz_sonuclari.pose_landmarks.landmark
        burun = (int(lm[POSE_NOSE_IDX].x * w), int(lm[POSE_NOSE_IDX].y * h))
        sol_omuz = (int(lm[POSE_LEFT_SHOULDER_IDX].x * w), int(lm[POSE_LEFT_SHOULDER_IDX].y * h))
        sag_omuz = (int(lm[POSE_RIGHT_SHOULDER_IDX].x * w), int(lm[POSE_RIGHT_SHOULDER_IDX].y * h))
        oran = calculate_head_drop_ratio(burun, sol_omuz, sag_omuz)
        for tid, kutu in takip_kutulari.items():
            if kutu[0] <= burun[0] <= kutu[2] and kutu[1] <= burun[1] <= kutu[3]:
                ihlal_yoneticisi.update_head_state(tid, oran, su_anki_sn)
                break

def videoyu_modelle_analiz_et(video_yolu, model_anahtari, izleme_bolgeleri=None, kvkk_modu=False):
    alarm.durdur()  # Analiz başlarken alarmı sıfırla
    yapilandirma = MODEL_YAPILANDIRMALARI[model_anahtari]
    model = YOLO(yapilandirma["engine"]) if os.path.exists(yapilandirma["engine"]) else YOLO(yapilandirma["pt"])
    yakalama = cv2.VideoCapture(video_yolu)
    video_alani = st.empty()
    takipci = CentroidTracker(max_distance=IOU_TAKIP_ESIGI)
    
    # config.json'dan güncel değerleri al
    s_sec = ayarlar.get("stillness_seconds", 6.0)
    e_sec = ayarlar.get("eye_closed_seconds", 6.0)

    ihlal_yoneticisi = ViolationManager(
        still_threshold=s_sec, eye_threshold=e_sec,
        movement_threshold=HAREKET_PIKSEL_ESIGI, ear_threshold=EAR_ESIGI
    )
    yuz_agi = mp_face_mesh.FaceMesh(refine_landmarks=True)
    fps_video = yakalama.get(cv2.CAP_PROP_FPS); toplam_kare = int(yakalama.get(cv2.CAP_PROP_FRAME_COUNT))
    kare_indeksi = 0; toplam_isleme_suresi = 0.0; islenen_kare_sayisi = 0
    ilerleme_cubugu = st.progress(0)

    try:
        while True:
            durum, cerceve = yakalama.read()
            if not durum: break
            su_anki_sn = kare_indeksi / fps_video if fps_video > 0 else 0.0
            kare_indeksi += 1; baslangic_zamani = time.time()

            yolo_sonuclari = model(cerceve, conf=GUVEN_ESIGI, classes=[0], device=CIHAZ, verbose=False)
            tespitler = []
            if len(yolo_sonuclari) > 0:
                for kutu in yolo_sonuclari[0].boxes:
                    x1, y1, x2, y2 = map(int, kutu.xyxy[0])
                    if izleme_bolgeleri:
                        if not any(is_point_in_rect(((x1+x2)//2, (y1+y2)//2), z) for z in izleme_bolgeleri): continue
                    tespitler.append((x1, y1, x2, y2))

            takip_kutulari = takipci.update(tespitler)
            ihlal_yoneticisi.update_tracks(takip_kutulari, su_anki_sn)
            cerceve_rgb = cv2.cvtColor(cerceve, cv2.COLOR_BGR2RGB)
            yuz_sonuclari = yuz_agi.process(cerceve_rgb)
            if yuz_sonuclari.multi_face_landmarks:
                ear_hesapla_ve_bulaniklastir(cerceve, yuz_sonuclari.multi_face_landmarks, takip_kutulari, ihlal_yoneticisi, su_anki_sn, kvkk_modu)
            poz_sonuclari = poz_dedektoru.process(cerceve_rgb)
            poz_verisini_isle(cerceve, poz_sonuclari, takip_kutulari, ihlal_yoneticisi, su_anki_sn)

            is_global_crit, ham_nedenler, kisi_basi = ihlal_yoneticisi.compute_violations(su_anki_sn)
            
            # TELEGRAM & ALARM FİLTRELEME
            if "TELEGRAM_BILDIRIMI_GEREKLI" in ham_nedenler:
                ham_nedenler.remove("TELEGRAM_BILDIRIMI_GEREKLI")
                for tid, state in ihlal_yoneticisi.person_states.items():
                    if not state.telegram_sent:
                        durum_mesaji = ""
                        if state.eye_violation_active and (su_anki_sn - state.eye_closed_start_time >= 30.0):
                            durum_mesaji = f"🔴 KRITIK: ID={tid} 30 saniyedir UYUYOR!"
                        elif state.still_violation_active and (su_anki_sn - state.still_start_time >= 30.0):
                            durum_mesaji = f"🟡 UYARI: ID={tid} 30 saniyedir HAREKETSIZ!"
                        
                        if durum_mesaji:
                            telegram_foto_gonder(cerceve.copy(), durum_mesaji)
                            state.telegram_sent = True

            goruntuleme_durumu = ""; durum_rengi = (0, 255, 0)
            for tid, p_durumu in kisi_basi.items():
                if p_durumu["eye_violation"]:
                    goruntuleme_durumu = "UYUYOR"; durum_rengi = (0, 0, 255)
                    break
                elif p_durumu["still_violation"]:
                    goruntuleme_durumu = "HAREKETSIZ"; durum_rengi = (0, 255, 255)
                    break

            kisi_bbox_ciz(
                cerceve, takip_kutulari, ihlal_yoneticisi, kisi_basi,
                s_sec, e_sec, EAR_ESIGI,
            )

            if goruntuleme_durumu:
                cv2.putText(cerceve, f"DURUM: {goruntuleme_durumu}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, durum_rengi, 3, cv2.LINE_AA)
                alarm.guncelle(ham_nedenler)
            else:
                alarm.guncelle([])

            if izleme_bolgeleri:
                for z in izleme_bolgeleri: cv2.rectangle(cerceve, (z[0], z[1]), (z[2], z[3]), (0, 255, 255), 2)

            gecikme = (time.time() - baslangic_zamani) * 1000
            anlik_fps = 1.0 / (time.time() - baslangic_zamani) if (time.time() - baslangic_zamani) > 0 else 0
            cv2.putText(cerceve, f"FPS: {anlik_fps:.1f} | {gecikme:.1f}ms", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            islenen_kare_sayisi += 1; toplam_isleme_suresi += (time.time() - baslangic_zamani)
            if toplam_kare > 0: ilerleme_cubugu.progress(min(1.0, kare_indeksi / toplam_kare))
            video_alani.image(cv2.cvtColor(cerceve, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        ihlal_yoneticisi.finalize(su_anki_sn)
    finally:
        alarm.durdur()
        yakalama.release(); yuz_agi.close(); ilerleme_cubugu.empty()
    return ihlal_yoneticisi.episodes, islenen_kare_sayisi / toplam_isleme_suresi if toplam_isleme_suresi > 0 else 0.0

# ===========================================================
# RAPORLAMA & DASHBOARD
# ===========================================================

def raporu_csv_kaydet(model_sonuclari):
    os.makedirs(CIKTI_RAPOR_DIZINI, exist_ok=True)
    dosya_yolu = os.path.join(CIKTI_RAPOR_DIZINI, f"Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    veri_listesi = []
    for ma, veri in model_sonuclari.items():
        for ep in veri["episodes"]:
            veri_listesi.append({"model": MODEL_YAPILANDIRMALARI[ma]['label'], "ihlal_turu": ep.violation_type, "baslangic": ep.start_sec, "bitis": ep.end_sec, "sure": ep.duration})
    if veri_listesi: pd.DataFrame(veri_listesi).to_csv(dosya_yolu, index=False, sep=';', encoding='utf-8-sig')
    return dosya_yolu

def paneli_olustur():
    st.subheader("📊 Günlük Özet & Analitik Dashboard")
    rapor_dosyalari = glob.glob(os.path.join(CIKTI_RAPOR_DIZINI, "Rapor_*.csv"))
    if not rapor_dosyalari:
        st.info("Henüz analiz verisi bulunamadı.")
        return

    tum_veriler = []
    for f in rapor_dosyalari:
        try:
            gecici_df = pd.read_csv(f, sep=';', encoding='utf-8-sig')
            if 'ihlal' in gecici_df.columns: gecici_df = gecici_df.rename(columns={'ihlal': 'ihlal_turu'})
            dosya_adi = os.path.basename(f)
            parcalar = dosya_adi.split("_")
            if len(parcalar) >= 3:
                tarih_metni = parcalar[1]
                saat_metni = parcalar[2]
                gecici_df['tarih'] = pd.to_datetime(tarih_metni, format='%Y%m%d')
                gecici_df['saat'] = int(saat_metni[:2])
            tum_veriler.append(gecici_df)
        except: continue
    
    if not tum_veriler: return
    df = pd.concat(tum_veriler, ignore_index=True)
    
    # Metrikler
    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam İhlal Sayısı", len(df))
    m2.metric("Ortalama İhlal Süresi", f"{df['sure'].mean():.2f} sn")
    if not df.empty and 'saat' in df.columns:
        zirve_saat = df['saat'].mode()[0]
        m3.metric("En Yoğun Saat Dilimi", f"{zirve_saat}:00 - {zirve_saat+1}:00")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("🕒 **Saatlik İhlal Dağılımı**")
        saatlik_df = df.groupby('saat').size().reset_index(name='sayi')
        st.plotly_chart(px.bar(saatlik_df, x='saat', y='sayi', color_discrete_sequence=['#ff4b4b']), use_container_width=True)

    with c2:
        st.write("📈 **Günlük İhlal Trendi**")
        trend_df = df.groupby('tarih').size().reset_index(name='sayi')
        st.plotly_chart(px.line(trend_df, x='tarih', y='sayi', markers=True), use_container_width=True)

    st.markdown("---")
    st.write("🤖 **Model Karşılaştırması (İhlal Türüne Göre)**")
    model_df = df.groupby(['model', 'ihlal_turu']).size().reset_index(name='sayi')
    st.plotly_chart(px.bar(model_df, x='model', y='sayi', color='ihlal_turu', barmode='group'), use_container_width=True)

# ===========================================================
# MAIN & ALARM
# ===========================================================

def main():
    st.set_page_config(page_title="GuardWatch AI", layout="wide", page_icon="🌙")
    st.title("🌙 GuardWatch AI: Otonom Güvenlik Takip Sistemi")
    tablo1, tablo2 = st.tabs(["🎥 Video Analizi", "📊 Analitik Dashboard"])
    with tablo1:
        sutun1, sutun2 = st.columns([1, 3])
        with sutun1:
            st.subheader("🛠️ Ayarlar")
            ayarlar['alarm_settings']['enabled'] = st.toggle("🚨 Alarm Sesleri", value=ayarlar['alarm_settings']['enabled'])
            roi_kullan = st.toggle("🎯 Bölge İzleme (ROI) Aktif", value=False)
            kvkk_modu = st.checkbox("🔒 KVKK Gizlilik Modu", value=True)
            secilen_modeller = st.multiselect("Modeller", options=list(MODEL_YAPILANDIRMALARI.keys()), default=[list(MODEL_YAPILANDIRMALARI.keys())[0]], format_func=lambda k: MODEL_YAPILANDIRMALARI[k]["label"])
            yuklenen_dosya = st.file_uploader("📂 Video Dosyası", type=["mp4"])
            baslat_btn = st.button("🚀 Analizi Başlat")
        with sutun2:
            if yuklenen_dosya and baslat_btn:
                gecici_yol = "temp_video.mp4"
                with open(gecici_yol, "wb") as f: f.write(yuklenen_dosya.getbuffer())
                izleme_bolgeleri = videodan_bolgeleri_sec(gecici_yol) if roi_kullan else None
                sonuclar = {}
                for ma_anahtar in secilen_modeller:
                    bolumler, fps = videoyu_modelle_analiz_et(gecici_yol, ma_anahtar, izleme_bolgeleri=izleme_bolgeleri, kvkk_modu=kvkk_modu)
                    sonuclar[ma_anahtar] = {"episodes": bolumler, "fps": fps}
                raporu_csv_kaydet(sonuclar)
                st.success("Analiz tamamlandı.")
            else: st.info("Video yükleyerek analizi başlatın.")
    with tablo2: paneli_olustur()

if __name__ == "__main__": main()
