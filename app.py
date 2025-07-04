import streamlit as st
import cv2
from yolo_tracker import detect_and_track
import tempfile

st.title("🔍 Real-Time Object Detection & Tracking")

source = st.radio("Choose input source:", ["Webcam", "Upload Video"])

if source == "Upload Video":
    video_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
elif source == "Webcam":
    video_path = 0  # webcam

start = st.button("Start Detection")

if start:
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracks = detect_and_track(frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = track.track_id
            label = f"ID {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
