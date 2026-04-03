import cv2
import mediapipe as mp
import math
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(x1, y1, x2, y2):
    angle_radian = math.atan2(y2 - y1, x2 - x1)
    angle_degree = math.degrees(angle_radian)
    return angle_degree


def analyze_swing_with_score(max_shoulder_turn, max_hip_turn, center_shift, finish_shoulder_angle, finish_hand_y, shoulder_y):
    results = []
    score = 100

    # 1. 轉體不足
    if max_shoulder_turn < 40 or max_hip_turn < 25:
        results.append({
            "problem": "轉體不足",
            "advice": "建議加強上桿時肩膀與髖部的旋轉，提升蓄力與揮桿完整性。"
        })
        score -= 25

    # 2. 重心不穩
    if center_shift > 40:   # 這裡是像素值，不是小數
        results.append({
            "problem": "重心不穩",
            "advice": "建議練習下盤穩定與核心控制，減少揮桿時左右晃動。"
        })
        score -= 25

    # 3. 收桿不完整
    if finish_shoulder_angle < 70:
        results.append({
            "problem": "收桿不完整",
            "advice": "建議擊球後延續轉體，讓身體自然完成收桿。"
        })
        score -= 25

    # 4. 手部過低
    if finish_hand_y > shoulder_y:
        results.append({
            "problem": "手部過低",
            "advice": "建議收桿時提高雙手位置，讓手部接近肩膀或更高。"
        })
        score -= 25

    if not results:
        results.append({
            "problem": "動作正常",
            "advice": "目前未偵測到明顯異常，揮桿動作整體表現穩定。"
        })

    return results, score


# 建立 Pose 模型
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 讀取影片
cap = cv2.VideoCapture("golf3.mp4")

if not cap.isOpened():
    print("錯誤：無法開啟影片 golf3.mp4")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("analysis.mp4", fourcc, fps, (width, height))

# 初始化變數
frame_count = 0
max_diff = 0
max_frame = None
max_frame_index = 0

nose_x_list = []
wrist_shoulder_diff_list = []
last_diff = 0

max_shoulder_turn = 0
max_hip_turn = 0
center_positions = []
finish_shoulder_angle = 0
finish_hand_y = 0
shoulder_y = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = result.pose_landmarks.landmark
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

        # 鼻子 x 座標（重心左右位移參考）
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_x = int(nose.x * w)
        nose_x_list.append(nose_x)

        # 手腕與肩膀高度
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_wrist_y = int(lw.y * h)
        right_wrist_y = int(rw.y * h)

        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        avg_shoulder_y = (sy1 + sy2) / 2

        wrist_shoulder_diff = avg_wrist_y - avg_shoulder_y
        wrist_shoulder_diff_list.append(wrist_shoulder_diff)

        # ===== 更新分析用資料 =====
        max_shoulder_turn = max(max_shoulder_turn, abs(shoulder_angle))
        max_hip_turn = max(max_hip_turn, abs(hip_angle))
        center_positions.append(nose_x)
        finish_shoulder_angle = abs(shoulder_angle)
        finish_hand_y = avg_wrist_y
        shoulder_y = avg_shoulder_y

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
pose.close()

# 安全計算重心位移
if len(center_positions) > 0:
    center_shift = max(center_positions) - min(center_positions)
else:
    center_shift = 0

# 呼叫分析函式
analysis_results, score = analyze_swing_with_score(
    max_shoulder_turn,
    max_hip_turn,
    center_shift,
    finish_shoulder_angle,
    finish_hand_y,
    shoulder_y
)

# 印出分析數據
print("\n=== 基本數據 ===")
print(f"最大肩膀角度: {max_shoulder_turn:.2f}")
print(f"最大髖部角度: {max_hip_turn:.2f}")
print(f"重心位移: {center_shift:.2f}")
print(f"收桿肩膀角度: {finish_shoulder_angle:.2f}")
print(f"收桿手部高度: {finish_hand_y:.2f}")
print(f"肩膀高度: {shoulder_y:.2f}")

print("\n=== 分析結果 ===")
print(f"總分: {score}")

for r in analysis_results:
    print(f"{r['problem']}：{r['advice']}")

# 存最大差值畫面
if max_frame is not None:
    cv2.imwrite("max_frame.jpg", max_frame)
    print("\n已儲存最大差值畫面：max_frame.jpg")

print(f"最大差值: {max_diff:.2f}")
print(f"出現在第幾幀: {max_frame_index}")