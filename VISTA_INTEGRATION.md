# Radar Gate + VISTA olay-sonrasi arac dogrulama

Bu entegrasyon `radar_gate.py` icindeki START/FINISH piksel olcumunu degistirmez.
Ihlal fotografi kaydedildikten sonra ayri ve dusuk oncelikli bir process:

1. VISTA'nin YOLOX-s ONNX modeliyle arac arar.
2. Arac yoksa OCR'i hic calistirmadan fotografi varsayilan olarak
   `ihlaller/rejected/*_ARAC_YOK.jpg` adiyla tasir.
3. Arac varsa arac kutusunu ve alt bolgesini OCR adayi olarak kaydeder.
4. OCR'i sirasiyla alt arac bolgesi, tum arac ve son olarak asil fotograf uzerinde dener.
5. AI hata verirse fail-open davranir: fotografi korur ve mevcut OCR'i asil fotografa uygular.

## Ayarlar

Systemd servisindeki ortam degiskenleri:

- `RADAR_AI_MODEL`: ONNX model yolu.
- `RADAR_AI_THREADS`: AI process OpenCV thread sayisi; Pi 5 icin once `2`.
- `RADAR_AI_CONFIDENCE`: en dusuk arac guveni; baslangic icin `0.25`.
- `RADAR_AI_REJECT_MODE`: `move` (onerilen), `keep` veya `delete`.
- `RADAR_AI_NICE`: Linux dusuk oncelik degeri; `10` onerilir.
- `RADAR_OCR_SINGLE_MIN_CONFIDENCE`: tek OCR sonucunu kabul esigi; varsayilan `0.98`.

Ilk saha testlerinde `move` kullanin. Yanlis negatifler incelendikten sonra gerekirse
`delete` secilebilir.

Karar davranisi:

- `move`: arac olmayan fotografi `_ARAC_YOK` ekiyle `rejected/` altina tasir; OCR yok.
- `keep`: ayni klasorde `_ARAC_YOK` ekiyle korur; OCR yok.
- `delete`: aracsiz fotografi siler; OCR yok.
- Arac varsa asil ihlal dosyasi web/CSV baglantilari bozulmasin diye yeniden
  adlandirilmaz; AI metadata JSON'unda `ARAC_VAR_<SINIF>` etiketi bulunur ve OCR calisir.

## Ciktilar

- Onaylanan asil fotograflar: `ihlaller/`
- Reddedilen fotograflar: `ihlaller/rejected/`
- OCR icin uretilen crop'lar: `violations/vehicle_crops/`
- AI karar JSON'lari: `violations/ai_metadata/`
- AI logu: `violations/vehicle_validation.log`
- OCR sonuclari: `violations/violations.csv` ve `violations/violations.jsonl`
- OCR adaylari ve uzlasma karari: `violations/ocr_review/`

OCR, alt arac crop'u, tam arac crop'u ve asil fotografi birlikte dener. Normal
guven seviyesinde en az iki crop ayni plakayi vermedikce sonucu kesin plaka olarak
kaydetmez; celiskili adaylar `OKUNAMADI` olarak isaretlenir ve inceleme JSON'unda korunur.

Desteklenen plaka bicimleri birbirinden ayridir:

- Standart TR: il kodu + 1-3 harf + 2-4 rakam (ornegin `34ABC123`).
- Ozel sayisal format: tam 6 rakam (ornegin `012345`); bastaki sifir korunur.

Alti haneli sayisal adaylarda harf/rakam karakter duzeltmesi yapilmaz ve tek crop sonucu kabul
edilmez. En az iki OCR girdisi ayni 6 rakami vermelidir. Bu nedenle ekran tarihi,
hiz yazisi veya yol numarasi plaka diye kaydedilmez; standart plaka yolu da degismez.

## Model testi

OCR veya servis baslatmadan tek fotograf testi:

```bash
cd /home/radarpi/radar
python3 vehicle_validation_worker.py /home/radarpi/radar/test-arac.jpg
```

JSON icinde `status: vehicle_confirmed` ve secilen arac kutusu gorulmelidir.

## OCR kurulumu

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-eng
python3 -m pip install -r /home/radarpi/radar/requirements_ocr.txt
```

RapidOCR ana backend, Tesseract ise gercek fallback'tir. RapidOCR sonuc bulamazsa
kod artik Tesseract denemesine devam eder. RapidOCR sonucu `[box, text, confidence]`
sozlesmesine gore ayrıştırılır; kutu koordinatlari metin olarak yorumlanmaz.
