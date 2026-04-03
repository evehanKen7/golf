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

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("analysis.mp4", fourcc, fps, (width, height))

frame_count = 0
max_diff = 0
max_frame = None
max_frame_index = 0

nose_x_list = []
wrist_shoulder_diff_list = []
last_diff = 0

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
        last_diff = diff

        if diff > max_diff:
            max_diff = diff
            max_frame = frame.copy()
            max_frame_index = frame_count

        # 抓鼻子 x 座標（重心穩定參考）
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_x = int(nose.x * w)
        nose_x_list.append(nose_x)

        # 抓手腕與肩膀的高度差（手部過低參考）
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_wrist_y = int(lw.y * h)
        right_wrist_y = int(rw.y * h)

        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        avg_shoulder_y = (sy1 + sy2) / 2

        wrist_shoulder_diff = avg_wrist_y - avg_shoulder_y
        wrist_shoulder_diff_list.append(wrist_shoulder_diff)

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

    out.write(frame)
    cv2.imshow("Golf Swing Analysis", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 存最大差值畫面
if max_frame is not None:
    cv2.imwrite("max_frame.jpg", max_frame)

print("最大差值:", max_diff)
print("出現在第幾幀:", max_frame_index)

# -------------------------
# 四種判斷建議系統
# -------------------------

advice_dict = {
    "轉體不足": "可加強肩膀與髖部轉動訓練",
    "重心不穩": "練習固定下半身與核心穩定",
    "收桿不完整": "完成揮桿後保持平衡 2 秒",
    "手部過低": "注意上桿時手臂抬升路徑"
}

detected_results = []

# 1. 轉體不足
if max_diff < 5:
    detected_results.append("轉體不足")

# 2. 重心不穩
if len(nose_x_list) > 0:
    nose_movement = max(nose_x_list) - min(nose_x_list)
    if nose_movement > 40:
        detected_results.append("重心不穩")

# 3. 收桿不完整
if last_diff < 3:
    detected_results.append("收桿不完整")

# 4. 手部過低
if len(wrist_shoulder_diff_list) > 0:
    avg_wrist_shoulder_diff = sum(wrist_shoulder_diff_list) / len(wrist_shoulder_diff_list)
    if avg_wrist_shoulder_diff > 80:
        detected_results.append("手部過低")

print("\n=== 揮桿建議系統 ===")
if detected_results:
    for result in detected_results:
        print(f"偵測結果：{result}")
        print(f"建議內容：{advice_dict[result]}")
        print("-" * 30)
else:
    print("未偵測到明顯問題，整體動作表現穩定。")