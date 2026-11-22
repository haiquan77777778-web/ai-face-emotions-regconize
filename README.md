# Nhận diện biểu cảm khuôn mặt

Ứng dụng dùng duy nhất **MediaPipe Face Landmarker** để phát hiện/căn chỉnh khuôn mặt và **EfficientNet** để phân loại 7 biểu cảm: surprise, fear, disgust, happy, sad, angry, neutral.

## Chức năng

- Nhận diện trực tiếp từ webcam hoặc ảnh JPG, PNG, WebP.
- Phân tích tối đa 5 khuôn mặt và hiển thị xác suất 7 biểu cảm.
- Kiểm tra tư thế, khoảng cách, ánh sáng, độ mờ, nhiễu, khuôn mặt sát biên và độ tin cậy.
- Có công tắc **Kiểm tra chất lượng** để bật/tắt cảnh báo và hướng dẫn khắc phục.

## Cài đặt và chạy

```powershell
setup.bat
.\venv\Scripts\streamlit.exe run app.py
```

Hoặc cài thủ công:

```powershell
pip install -r requirements_full.txt
python check_sync.py
streamlit run app.py
```

## Thành phần chính

```text
app.py                      giao diện Streamlit
core/detectors.py           MediaPipe Face Landmarker
core/predict.py             suy luận EfficientNet
core/pipeline.py            detect, crop, predict, kiểm tra chất lượng
core/image_quality.py       các cảnh báo chất lượng ảnh
models/efficientnet.keras   trọng số mô hình
models/labels.txt           thứ tự 7 nhãn
```
