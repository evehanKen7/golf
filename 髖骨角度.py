import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("golf.mp4")

frame_count = 0

def calculate_angle(x1, y1, x2, y2):
    angle_radian = math.atan2(y2 - y1, x2 - x1)
    angle_degree = math.degrees(angle_radian)
    return angle_degree

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # 肩膀
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        sx1 = int(left_shoulder.x * w)
        sy1 = int(left_shoulder.y * h)
        sx2 = int(right_shoulder.x * w)
        sy2 = int(right_shoulder.y * h)

        shoulder_angle = calculate_angle(sx1, sy1, sx2, sy2)

        # 髖部
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        hx1 = int(left_hip.x * w)
        hy1 = int(left_hip.y * h)
        hx2 = int(right_hip.x * w)
        hy2 = int(right_hip.y * h)

        hip_angle = calculate_angle(hx1, hy1, hx2, hy2)

        # 畫肩膀線
        cv2.line(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)

        # 畫髖部線
        cv2.line(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 3)

        # 顯示角度
        cv2.putText(
            frame,
            f"Shoulder Angle: {shoulder_angle:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Hip Angle: {hip_angle:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        if frame_count % 10 == 0:
            print(f"Frame {frame_count}")
            print(f"Shoulder Angle: {shoulder_angle:.2f}")
            print(f"Hip Angle: {hip_angle:.2f}")
            print("-" * 30)

    cv2.imshow("Golf Swing Analysis", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()