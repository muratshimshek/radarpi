# SpeedCam V2

Raspberry Pi 5 + CSI kamera icin yeni, moduler hiz tespit omurgasi.

## Ne var

- `main.py`: runtime entrypoint
- `camera.py`: Picamera2 veya OpenCV capture thread
- `detector.py`: NCNN primary, OpenCV fallback
- `tracker.py`: Kalman prediction + custom association
- `line_crossing.py`: bottom-center ile sari/kirmizi gecis
- `speed_estimator.py`: state machine ve hiz hesabi
- `homography.py`: homography kaydet/yukle/projeksiyon
- `capture.py`: en net frame secimi ve crop kaydi
- `ui.py`: overlay HUD
- `webui.py`: mevcut projeye yakin Flask arayuzu
- `configs/default.yaml`: varsayilan saha ayarlari

## Pi kurulumu

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-libcamera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_pi.txt
```

NCNN Python binding kullaniyorsan onu ayrica kur.

Hazir resmi YOLO ile hizli test yapmak istersen:

```bash
mkdir -p /home/radarpi/radar/speedcam_v2/models
cd /home/radarpi/radar/speedcam_v2/models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

Sonra replay config `backend: ultralytics` ile acilabilir.

## Canli kullanim

Asil hedef bu dosyadir:

- `configs/live_pi_ultralytics.yaml`

Pi kamera ile canli calistirma:

```bash
cd /home/radarpi/radar
python3 -m speedcam_v2.main --config /home/radarpi/radar/speedcam_v2/configs/live_pi_ultralytics.yaml
```

Web arayuzu:

- `http://<pi-ip>:5001`

Bu modda:

- Picamera2 kullanilir
- hazir YOLOv8n modeli ile `car` tespiti yapilir
- OpenCV fallback kapatilir
- sari/kirmizi cizgiler sahada ayarlanir

## Replay

```bash
cd /home/radarpi/radar/speedcam_v2
python3 -m speedcam_v2.main
```

## Replay testi

Gercek kameraya gecmeden once problemli videolari replay ile test et:

```bash
python3 -m speedcam_v2.main --config speedcam_v2/configs/replay_133556.yaml
```

Sadece pipeline testi yapmak istersen:

```bash
python3 -m speedcam_v2.main --config speedcam_v2/configs/replay_133556.yaml --no-web --max-runtime 20
```

## Kalibrasyon

Ilk fazda:

- `configs/default.yaml` icindeki `profiles.SAG` veya `profiles.SOL`
- `yellow_line`
- `red_line`
- `lane_polygon`
- `measurement.real_distance_meters`

degerlerini web arayuzu veya YAML ile ayarlayacaksin.

Canli kurulumda once replay degil, kamera acisini ac ve cizgileri gercek sahaya gore ayarla.

Homography aktif etmek icin:

- `homography.enabled: true`
- `image_points`
- `world_points`

gir ve kaydet.

## Egitim

PC tarafinda:

```bash
cd speedcam_v2/training
pip install -r ../requirements_pc.txt
python train.py --dataset C:/path/to/dataset --epochs 120 --imgsz 640
```

## NCNN export

```bash
cd speedcam_v2/training
python export_ncnn.py --weights runs/detect/train/weights/best.pt
```

Export sonrasi olusan modeli Pi'de `models/yolov8n_car_ncnn_model` altina koy.

## Notlar

- Ilk surumde NCNN decode katmani export formatina gore tamamlanacak.
- Model hazir degilse veya NCNN binding kurulu degilse OpenCV fallback devreye girer.
- Production hedefinde fallback kapatilabilir.
- Windows masaustu denemesinde `opencv-python` yoksa replay acilmaz; Pi veya OpenCV kurulu bir Python ile test et.
- Hazir resmi YOLO testinde COCO `car` sinifi kullanilir; bu sinif id'si config'te `2` olarak ayarlidir.

## Servis

Hazir systemd unit ornegi:

- `deploy/speedcam-v2.service`

Pi'ye kopyalayip etkinlestirmek icin:

```bash
sudo cp /home/radarpi/radar/speedcam_v2/deploy/speedcam-v2.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable speedcam-v2.service
sudo systemctl start speedcam-v2.service
sudo systemctl status speedcam-v2.service --no-pager
```
