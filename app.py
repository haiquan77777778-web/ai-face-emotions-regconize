"""
app.py - Giao diện Streamlit cho hệ thống nhận diện biểu hiện khuôn mặt


"""

import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import Pipeline
from preprocessing import prepare_face
from image_quality import analyze_frame_quality
from ui_helpers import draw_results, filter_results, open_camera


# ---------------------------------------------------------------- Cấu hình cố định
detector = "mediapipe"
model_name = "efficientnet"
CROP_MARGIN = 0.15                             # cố định, không hiển thị trên giao diện
SMOOTH_N = 3                                   # cố định, không hiển thị trên giao diện
PREVIEW_CROP = False                           # tắt phần xem trước ảnh crop

st.set_page_config(page_title="Nhận diện biểu hiện khuôn mặt", page_icon="🙂", layout="wide")
st.title("Hệ thống nhận diện biểu hiện khuôn mặt")

with st.sidebar:
    st.header("Cấu hình")
    st.caption("Phát hiện: MediaPipe · Mô hình: EfficientNet")
    align_faces = st.toggle("Căn chỉnh khuôn mặt", value=True)
    check_quality = st.toggle(
        "Kiểm tra chất lượng",
        value=True,
        help="Bật cảnh báo tư thế, khoảng cách, ánh sáng, độ mờ, nhiễu, che khuất và hướng dẫn khắc phục.",
    )
threshold = 0.0                                # không lọc theo ngưỡng tin cậy


# ---------------------------------------------------------------- Trạng thái phiên
if "live_pipe" not in st.session_state:
    st.session_state.live_pipe = Pipeline()    # Live: phát hiện cách vài khung + làm mượt (mặc định của Người 3)


def run(pipe, frame_bgr):
    """Gọi pipeline và trả ảnh đã vẽ, kết quả khuôn mặt, chất lượng toàn ảnh."""
    pipe.align = align_faces
    pipe.crop_margin = CROP_MARGIN
    pipe.smooth_n = SMOOTH_N
    pipe.check_quality = check_quality
    results = filter_results(pipe.process_frame(frame_bgr, detector, model_name), threshold)
    frame_quality = analyze_frame_quality(frame_bgr) if check_quality else None
    return draw_results(frame_bgr, results), results, frame_quality


def all_warnings(results, frame_quality):
    warnings = list((frame_quality or {}).get("warnings", []))
    for i, result in enumerate(results, 1):
        for warning in (result.get("quality") or {}).get("warnings", []):
            warnings.append({**warning, "face": i})
    if not results:
        warnings.append({
            "code": "face_missing", "title": "Không phát hiện khuôn mặt",
            "guide": "Đưa toàn bộ khuôn mặt vào giữa khung hình, nhìn thẳng và bỏ vật che trước mặt."
        })
    return warnings


def warning_text(results, frame_quality):
    lines, seen = [], set()
    for warning in all_warnings(results, frame_quality):
        key = (warning["code"], warning.get("face"))
        if key in seen:
            continue
        seen.add(key)
        face = f"Khuôn mặt {warning['face']}: " if warning.get("face") else ""
        lines.append(f"- **{face}{warning['title']}** — {warning['guide']}")
    return "\n".join(lines)


def show_results(results):
    if not results:
        st.warning("Không phát hiện khuôn mặt, hoặc không có kết quả vượt ngưỡng tin cậy.")
        return
    for i, r in enumerate(results, 1):
        st.write(f"**Khuôn mặt {i}:** {r['label']} - {r['prob']:.1%}")
        quality = r.get("quality")
        if quality:
            if quality.get("ok", True):
                st.success("Chất lượng khuôn mặt đạt yêu cầu.")
            else:
                messages = [f"**{w['title']}** — {w['guide']}" for w in quality.get("warnings", [])]
                st.warning("\n\n".join(messages))
        probs = r.get("probs") or {}
        if probs:
            st.bar_chart(pd.Series(probs, name="Xác suất"), height=200)


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 2 tab
tab_live, tab_image = st.tabs(["Live", "Phân tích ảnh"])

# ---- Tab Live (vòng lặp OpenCV)
with tab_live:
    st.subheader("Nhận diện trực tiếp từ webcam")
    c1, c2, _ = st.columns([1, 1, 8], gap="small")
    if c1.button("▶ Bắt đầu", type="primary"):
        st.session_state.live = True
    if c2.button("■ Dừng"):
        st.session_state.live = False
    view, status, quality_box, fps_box, crop_box = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()

    if st.session_state.get("live", False):
        st.session_state.live_pipe.reset()
        cap = open_camera(0)
        if cap is None:
            st.session_state.live = False
            status.error("Không mở được webcam. Kiểm tra quyền camera hoặc đóng ứng dụng khác đang dùng camera.")
        else:
            status.info("Đang chạy. Đổi cấu hình ở thanh bên sẽ khởi động lại vòng lặp. Nhấn Dừng để kết thúc.")
            n, t0, bad = 0, time.perf_counter(), 0
            try:
                while st.session_state.get("live", False):
                    ok, frame = cap.read()
                    if not ok:
                        bad += 1
                        if bad > 30:                               # mất khung liên tục thì mới dừng
                            status.warning("Không đọc được khung hình từ webcam.")
                            break
                        continue
                    bad = 0
                    frame = cv2.flip(frame, 1)                       # lật gương cho tự nhiên
                    try:
                        annotated, results, frame_quality = run(st.session_state.live_pipe, frame)
                    except Exception as exc:
                        status.error(f"Lỗi pipeline: {exc}")
                        break
                    view.image(rgb(annotated))
                    warnings = warning_text(results, frame_quality) if check_quality else ""
                    if not check_quality:
                        quality_box.empty()
                    elif warnings:
                        quality_box.warning("**Chất lượng hình ảnh chưa đảm bảo để nhận diện biểu cảm**\n\n" + warnings)
                    else:
                        quality_box.success("Chất lượng hình ảnh đạt yêu cầu.")
                    if PREVIEW_CROP:
                        if results:
                            largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
                            pipe = st.session_state.live_pipe
                            crop = prepare_face(pipe.get_face_crop(frame, largest["box"])).astype(np.uint8)
                            crop_box.image(crop, caption="Ảnh RGB 100×100 đưa vào mô hình (mặt lớn nhất)", width=220)
                        else:
                            crop_box.empty()
                    n += 1
                    fps_box.metric("FPS", f"{n / max(time.perf_counter() - t0, 1e-6):.1f}")
            finally:
                cap.release()
    else:
        status.info("Nhấn **Bắt đầu** để chạy webcam.")

# ---- Tab Phân tích ảnh tải lên
with tab_image:
    st.subheader("Tải ảnh lên để phân tích")
    st.caption("Hỗ trợ JPG, JPEG, PNG và WebP; có thể nhận diện tối đa 5 khuôn mặt trong một ảnh.")
    uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Không đọc được ảnh. Hãy chọn lại một file JPG, PNG hoặc WebP hợp lệ.")
        elif st.button("Phân tích ảnh", type="primary", use_container_width=True):
            try:
                image_pipe = Pipeline(smooth_n=1, detect_every=1, align=align_faces,
                                      crop_margin=CROP_MARGIN, check_quality=check_quality)
                annotated, image_results, image_quality = run(image_pipe, image_bgr)
                st.session_state.image_analysis = {
                    "name": uploaded.name, "annotated": annotated,
                    "results": image_results, "quality": image_quality,
                    "detector": detector, "model": model_name, "check_quality": check_quality,
                }
            except Exception as exc:
                st.error(f"Không thể phân tích ảnh: {exc}")

    analysis = st.session_state.get("image_analysis")
    if analysis:
        st.image(rgb(analysis["annotated"]), caption=f"Kết quả: {analysis['name']}", use_container_width=True)
        if analysis.get("check_quality", True):
            warnings = warning_text(analysis["results"], analysis["quality"])
            if warnings:
                st.warning("**Chất lượng hình ảnh chưa đảm bảo để nhận diện biểu cảm**\n\n" + warnings)
            else:
                st.success("Chất lượng hình ảnh đạt yêu cầu.")
        show_results(analysis["results"])
