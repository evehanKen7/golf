import cv2
import mediapipe as mp

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

# 讀取影片
cap = cv2.VideoCapture("golf.mp4")

print("影片是否成功讀取：", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV 讀進來是 BGR，MediaPipe 要 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 姿態辨識
    results = pose.process(rgb_frame)

    # 如果有偵測到人體，就畫骨架
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 顯示畫面
    cv2.imshow("Golf Swing Analysis", frame)

    # 按 q 離開
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()