import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from inference import run_forgery_localization_pipeline
import base64

def show_video_small(video_path, width=420, height=240):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
            <video width="{width}" height="{height}" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="Deepfake Detection", layout="wide")

st.title("🕵️ Deepfake Video Detection & Forgery Localization")
st.write("Upload a video, run detection, and view suspicious frames with explanation.")

# -------------------------------
# VIDEO UPLOAD
# -------------------------------
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    video_path = "temp_uploaded_video.mp4"

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.subheader("📽 Uploaded Video")

    show_video_small(video_path, width=420, height=240)



    # -------------------------------
    # RUN PREDICTION
    # -------------------------------
    if st.button("🚀 Run Deepfake Detection"):
        with st.spinner("Running model... Please wait..."):
            results = run_forgery_localization_pipeline(video_path)

        # -------------------------------
        # FINAL PREDICTION
        # -------------------------------
        pred_label = "FAKE" if results["prediction_class"] == 0 else "REAL"
        confidence = results["confidence"] * 100

        st.success(f"### 🧠 Final Prediction: **{pred_label}**")
        st.info(f"### 🔍 Confidence: **{confidence:.2f}%**")

        # -------------------------------
        # TEMPORAL GRAPH
        # -------------------------------
        # st.subheader("📈 Temporal Importance (Frame-wise)")

        # fig, ax = plt.subplots(figsize=(9, 3))
        # ax.plot(results["combined_temporal"], marker="o")
        # ax.set_xlabel("Frame Index")
        # ax.set_ylabel("Importance (0–1)")
        # ax.set_title("Combined Temporal Importance")
        # ax.grid(True)
        # st.pyplot(fig)

        # -------------------------------
        # SUSPICIOUS FRAMES
        # -------------------------------
        # st.subheader("🚨 Suspicious Frames")

        # suspicious_frames = sorted([
        #     f for f in os.listdir(".")
        #     if f.startswith("suspicious_frame_") and f.endswith(".png")
        # ])

        # if not suspicious_frames:
        #     st.warning("No suspicious frames detected.")
        # else:
        #     cols = st.columns(4)
        #     for idx, frame_path in enumerate(suspicious_frames):
        #         img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        #         cols[idx % 4].image(
        #             img,
        #             caption=frame_path,
        #             width=250   # ✅ replaced deprecated use_column_width
        #         )

        # -------------------------------
        # CLEANUP
        # -------------------------------
        if st.button("🧹 Clear Results"):
            for f in suspicious_frames:
                os.remove(f)
            st.rerun()
