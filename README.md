# 🌙 GuardWatch AI: Otonom Güvenlik Takip Sistemi

Bu proje, iş yerlerinde ve güvenlik noktalarında personelin uyku durumunu, göz kapalılığını ve hareketsizliğini takip eden, modüler ve profesyonel bir yapay zeka çözümüdür. **GuardWatch AI** entegrasyonu ile sadece bir izleme aracı değil, aynı zamanda otonom bir bildirim sistemidir.

## 🚀 Öne Çıkan Özellikler

### 1. 🤖 GuardWatch AI: Acil Durum Telegram Botu
- **Otonom Bildirim:** Sistem personelin 60 saniye boyunca aralıksız şekilde kritik uyku pozisyonunda (Kafa düşük + Gözler kapalı) kaldığını tespit ettiğinde devreye girer.
- **Visual Proof (Görsel Kanıt):** İhlalin kesinleştiği o anda kamera görüntüsünden snapshot alır ve anlık olarak yöneticiye (Patron) gönderir.
- **Anti-Spam (Flood Koruması):** Yöneticiye mesaj yağmuru gitmemesi için "Flag" mimarisi kullanır. Personel uyanıp durum normale dönene kadar ikinci bir mesaj kesinlikle atılmaz.

### 2. 🔒 KVKK & Gizlilik Modu (Face Blurring)
- **Yüz Bulanıklaştırma:** Personel gizliliğini korumak için gerçek zamanlı yüz mozaikleme özelliği.
- **Akıllı Snapshot:** Telegram'a gönderilen kanıt fotoğrafları da KVKK modu açıksa otomatik olarak bulanıklaştırılmış şekilde iletilir.

### 3. 📊 Gelişmiş Analitik Dashboard (Streamlit)
- **Günlük Özet:** Toplam ihlal sayısı, ortalama ihlal süresi ve en yoğun saat dilimi (Peak Hour) gibi kritik metrikler.
- **Zaman Serisi Analizi:** Saatlik dağılım (Bar Chart) ve günlük trend (Line Chart) grafikleri.
- **Model Karşılaştırma:** YOLO11n ve YOLO11s modellerinin performans ve tespit başarılarını karşılaştıran gruplandırılmış grafikler.

### 4. 🎯 Dinamik Bölge İzleme (ROI)
- **Odaklı Takip:** Kullanıcı video başında mouse ile izlemek istediği kritik bölgeleri seçebilir.
- **Filtreleme:** Seçili bölgelerin dışındaki hareketler ve kişiler analiz dışı bırakılarak yanlış alarmlar (False Positive) minimize edilir.

### 5. ⚡ Hiyerarşik Durum Yönetimi
- Ekrandaki bilgi kirliliğini önlemek için durumlar önem sırasına göre gösterilir:
  1. `KESİN UYUYOR!` (Kırmızı - Kritik Seviye & Telegram Bildirimi)
  2. `MESGUL (Dikkat Dagildi)` (Sarı - Orta Seviye)
  3. `Göz Kapalı / Hareketsizlik` (Sarı - Başlangıç Seviyesi)

### 6. 🛠️ Teknik Altyapı
- **Engines:** YOLO11 (Detection), MediaPipe FaceMesh (EAR Analysis), MediaPipe Pose (Head Drop).
- **Optimization:** TensorRT FP16 desteği ile düşük gecikme ve yüksek FPS.
- **Logging:** Günlük bazda detaylı olay ve hata kayıtları (`logs/`).

## 🛠️ Kurulum ve Çalıştırma

1. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Sistemi Başlatın:**
   * **Seçenek A — Web Dashboard (Streamlit):**
     ```bash
     streamlit run model_karsilastirma_app.py
     ```
     * **Video Analizi:** Tab 1 üzerinden video yükleyip modelleri karşılaştırın.
     * **Dashboard:** Tab 2 üzerinden geçmiş verileri grafiklerle inceleyin.

   * **Seçenek B — Doğrudan Canlı Kamera / Video Takip Motoru:**
     ```bash
     python uyku_takip.py
     ```
     * Webcam veya video üzerinden anlık uyku, göz kapalılığı, kafa düşmesi ve Telegram uyarılarını çalıştırır.

## ⚙️ Konfigürasyon (`config.json`)
Eşik değerleri, alarm seslerini ve model yollarını kod değiştirmeden bu dosya üzerinden güncelleyebilirsiniz:
- `stillness_seconds`: Hareketsizlik limiti.
- `eye_closed_seconds`: Göz kapalılık limiti.
- `ear_threshold`: Göz hassasiyeti.
- `alarm_settings`: Ses dosyaları ve cooldown süreleri.

---
*Bu sistem, endüstriyel güvenlik standartları ve KVKK uyumluluğu gözetilerek geliştirilmiştir.*
