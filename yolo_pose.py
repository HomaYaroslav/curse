from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

# Історія точок для трекінгу (person_id -> {keypoint_id -> [positions]})
track_history = defaultdict(lambda: defaultdict(list))
TRAIL_LENGTH = 30  # скільки кадрів зберігати слід

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Які точки відстежувати (можна змінити)
TRACKED_KEYPOINTS = [9, 10, 15, 16]  # зап'ястки + щиколотки

TRAIL_COLORS = [
    (0, 255, 255),    # жовтий - left wrist
    (255, 0, 255),    # рожевий - right wrist
    (0, 128, 255),    # помаранчевий - left ankle
    (255, 128, 0),    # блакитний - right ankle
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Трекінг через YOLO (persist=True зберігає ID між кадрами)
    results = model.track(frame, persist=True, verbose=False)

    annotated_frame = results[0].plot()

    for result in results:
        if result.keypoints is None:
            continue
        
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()
        keypoints = result.keypoints.xy.cpu().numpy()      # [N, 17, 2]
        confidences = result.keypoints.conf.cpu().numpy()  # [N, 17]

        for person_idx, track_id in enumerate(track_ids):
            person_kps = keypoints[person_idx]   # [17, 2]
            person_conf = confidences[person_idx] # [17]

            for color_idx, kp_id in enumerate(TRACKED_KEYPOINTS):
                x, y = person_kps[kp_id]
                conf = person_conf[kp_id]

                # Ігноруємо якщо точка не виявлена (conf < 0.5 або x,y == 0)
                if conf < 0.5 or (x == 0 and y == 0):
                    continue

                history = track_history[track_id][kp_id]
                history.append((int(x), int(y)))

                if len(history) > TRAIL_LENGTH:
                    history.pop(0)

                # Малюємо слід із затуханням
                color = TRAIL_COLORS[color_idx % len(TRAIL_COLORS)]
                for i in range(1, len(history)):
                    alpha = i / len(history)  # від 0 до 1
                    thickness = max(1, int(3 * alpha))
                    faded_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated_frame, history[i-1], history[i],
                             faded_color, thickness)

                # Крапка на поточній позиції
                cv2.circle(annotated_frame, (int(x), int(y)), 5, color, -1)

            # ID персони над головою
            nose_x, nose_y = person_kps[0]
            if nose_x > 0 and nose_y > 0:
                cv2.putText(annotated_frame, f'ID:{track_id}',
                            (int(nose_x) - 20, int(nose_y) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('YOLOv8 Pose Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()