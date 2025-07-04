from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")  # lightweight and fast
tracker = DeepSort()

def detect_and_track(frame):
    results = model(frame)[0]
    detections = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        detections.append(([x1, y1, x2, y2], conf, str(int(cls))))

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks
