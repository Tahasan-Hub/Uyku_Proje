# 🌙 AI Uyku ve Güvenlik Takip Sistemi

Bu proje, iş yerlerinde ve güvenlik noktalarında personelin uyku durumunu, göz kapalılığını ve hareketsizliğini takip eden, modüler ve profesyonel bir yapay zeka çözümüdür.

## 🚀 Temel Özellikler

### 1. Mesafe Bazlı Performans Analizi 
- `mesafe_testi.py` scripti ile modelin 1m, 3m, 5m, 7m ve 10m mesafelerindeki başarısı ölçülür.
- **Metrikler:** Kişi tespit oranı, EAR (Göz) başarı oranı ve FPS değerleri analiz edilir.
- **Raporlama:** Sonuçlar otomatik olarak tablo ve performans grafiği (% başarı) şeklinde sunulur.

### 2. Gelişmiş Alarm Sistemi 
- **Sesli Uyarı:** `pygame.mixer` ile ihlal türüne göre farklı ses tonları (Hareketsizlik: Bip, Göz Kapalı: Acil Siren).
- **Mute Özelliği:** Canlı takip sırasında `m` tuşu ile sesler anlık olarak kapatılıp açılabilir.
- **Cooldown:** Gereksiz ses kirliliğini önlemek için akıllı alarm bekleme süresi mekanizması.

### 3. Dinamik Yapılandırma - Config 
- Tüm sistem ayarları (Eşik değerler, model yolları, alarm ayarları) `config.json` dosyasından yönetilir.
- Kod değişikliği yapmadan sistem davranışını (EAR eşiği, ihlal süreleri vb.) değiştirebilirsiniz.

### 4. Profesyonel Log Sistemi 
- **Günlük Kayıt:** Her gün için `Log_YYYY-MM-DD.log` formatında ayrı dosyalar oluşturulur.
- **Seviyeli Loglama:** 
  - `INFO`: Sistem başlangıcı ve tespitler.
  - `WARNING`: İhlal başlangıcı (Süre sayımı).
  - `CRITICAL`: İhlal gerçekleşmesi ve görüntü kaydı.

### 5. Çoklu Bölge İzleme - Zone Monitoring 
- **Dinamik ROI:** Kullanıcı, mouse ile ekran üzerinde sadece izlemek istediği bölgeleri (Bölge A: Masa, Bölge B: Güvenlik Noktası) seçebilir.
- **Odaklı Takip:** Sistem sadece tanımlı bölgelerdeki kişileri analiz eder, dışındakileri yoksayarak verimliliği artırır.

### 6. Günlük Özet Dashboard - Streamlit 
- **Veri Görselleştirme:** `model_karsilastirma_app.py` üzerinden geçmiş raporların analizi.
- **Grafikler:** Plotly ve Matplotlib ile:
  - Saatlik İhlal Dağılımı (Bar Chart)
  - Günlük İhlal Trendi (Line Chart)
  - Model Performans Karşılaştırması (Grouped Bar Chart)

## 🛠️ Kurulum

1. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Modelleri hazırlayın:
   YOLO .pt veya .engine (TensorRT) dosyalarınızı `config.json` içinde tanımlayın.

## 📈 Kullanım

- **Canlı Kamera Takibi:** `python uyku_takip.py` (Açılışta bölge seçimi yapabilirsiniz).
- **Analitik Dashboard:** `streamlit run model_karsilastirma_app.py` (Video analiz edin ve geçmişi görün).
- **Performans Testi:** `python mesafe_testi.py` (Mesafe/Başarı grafiği üretin).

---
*Bu sistem, endüstriyel güvenlik standartları ve performans metrikleri göz önünde bulundurularak geliştirilmiştir.*
