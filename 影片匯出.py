import cv2
import mediapipe as mp
import math

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

# 取得影片資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 建立輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("analysis.mp4", fourcc, fps, (width, height))

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
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        sx1, sy1 = int(ls.x * w), int(ls.y * h)
        sx2, sy2 = int(rs.x * w), int(rs.y * h)

        shoulder_angle = calculate_angle(sx1, sy1, sx2, sy2)

        # 髖部
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        hx1, hy1 = int(lh.x * w), int(lh.y * h)
        hx2, hy2 = int(rh.x * w), int(rh.y * h)

        hip_angle = calculate_angle(hx1, hy1, hx2, hy2)

        # 差值
        diff = abs(shoulder_angle - hip_angle)

        # 畫線
        cv2.line(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
        cv2.line(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 3)

        # 顯示文字
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.2f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Hip: {hip_angle:.2f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Diff: {diff:.2f}",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 👉 寫入影片（最重要‼️）
    out.write(frame)

    cv2.imshow("Golf Swing Analysis", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # ⚠️一定要加
cv2.destroyAllWindows()