# RadarPi

Raspberry Pi 5 uzerinde iki sanal cizgideki piksel degisiminden gecis suresi ve
hiz hesaplayan cevrimdisi radar uygulamasi. Hiz limiti asildiginda fotograf
kaydeder; ayri bir arka plan sureci fotografi YOLOX ile dogrular. Arac yoksa
OCR calistirilmaz ve fotograf `ihlaller/rejected` altina tasinir. Arac varsa
OCR yalnizca daraltilmis arac/plaka bolgesinde calisir.

## Ana dosyalar

- `radar_gate.py`: aktif piksel tabanli radar ve web arayuzu
- `vehicle_validation_worker.py`: asenkron arac dogrulama ve OCR yonetimi
- `plate_reader.py`: standart ve alti haneli sayisal plaka okuma
- `models/yolox_s.onnx`: cevrimdisi arac algilama modeli
- `radarpi.service`: systemd servis tanimi

## Raspberry Pi 5 kurulumu

Projeyi `/home/radarpi/radar` dizinine kopyalayin ve:

```bash
cd /home/radarpi/radar
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ocr.txt
python -m py_compile radar_gate.py vehicle_validation_worker.py plate_reader.py
python test_vehicle_validation.py
```

Servisi kurmak icin:

```bash
sudo cp radarpi.service /etc/systemd/system/radarpi.service
sudo systemctl daemon-reload
sudo systemctl enable --now radarpi.service
systemctl status radarpi.service
```

Web arayuzu yerel agda `http://PI_ADRESI:5001` adresindedir.

## Gizlilik

Ihlal fotograflari, reddedilen kareler, OCR debug goruntuleri, loglar, yerel
ayarlar ve Python sanal ortami `.gitignore` ile depo disinda tutulur. Gercek
plaka veya saha goruntuleri kaynak depoya eklenmemelidir.

Ayrintili entegrasyon ve test bilgileri icin `VISTA_INTEGRATION.md` dosyasina
bakin.
