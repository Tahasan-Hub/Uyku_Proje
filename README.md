## Uyku Projesi

Bu proje iki ana bileşenden oluşur:

- **`uyku_takip.py`**: Canlı **webcam** üzerinden gerçek zamanlı uyku ve güvenlik takibi.
- **`model_karsilastirma_app.py`**: Önceden kaydedilmiş **.mp4 videoları** üzerinde iki YOLO modeli (yolo11n / yolo11s) ile performans ve ihlal karşılaştırma uygulaması (Streamlit arayüzlü).

---

### 1. Kurulum

Proje klasörüne geç:

```bash
cd "c:\Users\tahas\OneDrive\Desktop\Uyku Projesi"
```

Gerekli Python paketlerini yükle:

```bash
pip install ultralytics opencv-python mediapipe numpy streamlit
```

> Not: NVIDIA GPU, CUDA ve TensorRT sistemde kurulu olmalıdır. YOLO modelleri ilk çalıştırmada `.engine` (TensorRT FP16) formatına dönüştürülür, bu işlem birkaç dakika sürebilir.

---

### 2. Canlı Uyku ve Güvenlik Takibi (`uyku_takip.py`)

Çalıştırmak için:

```bash
python uyku_takip.py
```

Özellikler:

- `yolo11n.engine` ile **insan tespiti** ve her kişi için **benzersiz ID** takibi.
- **Hareketsizlik ihlali**: 10 saniye neredeyse sabit kalan kişi.
- **Goz Kapali ihlali**: MediaPipe FaceMesh + EAR ile 10 saniye göz kapalılığı.
- Her iki durumdan biri gerçekleştiğinde:
  - Ekranda büyük kırmızı uyarı (`UYKU IHLALI: ...`).
  - Anlık kare, tarih/saat ve ihlal nedeni ile birlikte `ihlal_kayitlari` klasörüne kaydedilir.
- Ekranda **FPS**, **EAR** ve her iki ihlal için geri sayım sayaçları gösterilir.

Çıkmak için pencere aktifken **`q`** tuşuna bas.

---

### 3. Model Karşılaştırma ve Raporlama (`model_karsilastirma_app.py`)

Streamlit arayüzünü başlatmak için:

```bash
streamlit run model_karsilastirma_app.py
```

Arayüz üzerinden:

1. Bilgisayarından bir **`.mp4` video** seç.
2. Analiz için **YOLO11n (nano)** ve/veya **YOLO11s (small)** modellerini işaretle.
3. **“Analizi Başlat”** butonuna tıkla.

Uygulama:

- Her model için videoyu baştan sona işler.
- 10 sn hareketsizlik ve 10 sn **Goz Kapali** ihlallerini **başlangıç / bitiş / süre (sn)** bilgileriyle tablo halinde gösterir.
- Her model için ortalama **FPS (işleme hızı)** hesaplar ve karşılaştırma tablosu sunar.
- Sonuçları proje içinde **`raporlar`** klasörüne, tarih-saat damgalı **CSV** dosyası olarak kaydeder.
- Aynı raporu arayüzden **“Raporu İndir (.csv)”** butonuyla bilgisayarına indirebilirsin.

Streamlit uygulamasını kapatmak için, çalıştığı PowerShell penceresinde **`Ctrl + C`** tuşlarına bas.

