# 🌙 AI Uyku ve Güvenlik Takip Sistemi

Bu proje, iş yerlerinde ve güvenlik noktalarında personelin uyku durumunu, göz kapalılığını ve hareketsizliğini takip eden profesyonel bir yapay zeka sistemidir.

## 🚀 Temel Özellikler

### 1. Canlı Takip & Analiz
- **YOLO11 & MediaPipe:** TensorRT optimize edilmiş YOLO11 modelleri ile yüksek FPS'li nesne tespiti ve MediaPipe FaceMesh ile milimetrik göz takibi.
- **Akıllı Alarm:** İhlal türüne göre (Göz Kapalılığı / Hareketsizlik) farklı ses tonlarıyla uyarı ve otomatik ihlal görüntüsü kaydı.

### 2. Çoklu Bölge İzleme (Zone Monitoring) - [GÖREV 5]
- **ROI Seçimi:** Kullanıcı ekran üzerinde fare ile belirli bölgeleri (masa, koltuk vb.) seçebilir.
- **Odaklı Takip:** Sistem sadece seçilen bölgelerdeki kişileri izler, dışındakileri yoksayarak hatalı alarmları önler.

### 3. Günlük Özet Dashboard - [GÖREV 6]
- **Veri Analitiği:** Streamlit arayüzü üzerinden geçmiş tüm analizlerin (`raporlar/` klasörü) otomatik özeti.
- **Görselleştirme:** Plotly ile Saatlik İhlal Dağılımı, Günlük Trend ve Model Karşılaştırma grafikleri.
- **Metrikler:** Toplam ihlal sayısı, ortalama ihlal süresi ve en yoğun çalışma saatleri analizi.

### 4. Mesafe Bazlı Performans Testi
- **Simülasyon:** Farklı fiziksel mesafelerdeki (1m - 10m) model başarısını ölçen dinamik test sistemi.
- **Grafiksel Rapor:** Mesafeye bağlı % başarı ve FPS değişimlerini gösteren profesyonel grafik çıktısı.

## 🛠️ Kurulum

1. Kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Yapılandırma:
   `config.json` üzerinden model yollarını ve eşik değerlerini düzenleyin.

## 📈 Kullanım

- **Kamera Takibi:** `python uyku_takip.py`
- **Dashboard & Video Analiz:** `streamlit run model_karsilastirma_app.py`
- **Performans Testi:** `python mesafe_testi.py`

---
*Bu proje profesyonel performans analizi ve iş yeri güvenliği için geliştirilmiştir.*
