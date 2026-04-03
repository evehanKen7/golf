import streamlit as st
import cv2
import mediapipe as mp
import math
import tempfile
import os
import subprocess
import imageio_ffmpeg

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
    if center_shift > 40:
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
def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return result.returncode == 0

def process_video(input_path, output_path, max_frame_path):
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps is None:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    max_diff = 0
    max_frame = None
    max_frame_index = 0

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

            if diff > max_diff:
                max_diff = diff
                max_frame = frame.copy()
                max_frame_index = frame_count

            # 鼻子 x 座標
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * w)
            center_positions.append(nose_x)

            # 手腕與肩膀高度
            lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_wrist_y = int(lw.y * h)
            right_wrist_y = int(rw.y * h)

            avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
            avg_shoulder_y = (sy1 + sy2) / 2

            # 更新分析資料
            max_shoulder_turn = max(max_shoulder_turn, abs(shoulder_angle))
            max_hip_turn = max(max_hip_turn, abs(hip_angle))
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

    cap.release()
    out.release()
    print("輸出影片路徑:", output_path)
    print("影片存在嗎:", os.path.exists(output_path))
    if os.path.exists(output_path):
        print("影片大小:", os.path.getsize(output_path))
    pose.close()

    if len(center_positions) > 0:
        center_shift = max(center_positions) - min(center_positions)
    else:
        center_shift = 0

    analysis_results, score = analyze_swing_with_score(
        max_shoulder_turn,
        max_hip_turn,
        center_shift,
        finish_shoulder_angle,
        finish_hand_y,
        shoulder_y
    )

    if max_frame is not None:
        cv2.imwrite(max_frame_path, max_frame)

    return {
        "max_shoulder_turn": max_shoulder_turn,
        "max_hip_turn": max_hip_turn,
        "center_shift": center_shift,
        "finish_shoulder_angle": finish_shoulder_angle,
        "finish_hand_y": finish_hand_y,
        "shoulder_y": shoulder_y,
        "max_diff": max_diff,
        "max_frame_index": max_frame_index,
        "analysis_results": analysis_results,
        "score": score
    }


st.set_page_config(page_title="高爾夫揮桿分析系統", layout="wide")

st.title("🏌️ 高爾夫揮桿 AI 分析系統")
st.write("上傳揮桿影片後，系統會自動分析骨架、判斷動作問題，並給出建議。")

uploaded_file = st.file_uploader("請上傳影片檔案", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.subheader("原始影片")
    st.video(uploaded_file)

    if st.button("開始分析"):
        with st.spinner("影片分析中，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                temp_input.write(uploaded_file.read())
                input_path = temp_input.name

                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            output_path = temp_output.name
            temp_output.close()

            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            max_frame_path = temp_image.name
            temp_image.close()

            result = process_video(input_path, output_path, max_frame_path)

            if result is None:
                st.error("影片讀取失敗，請確認檔案格式是否正確。")
            else:
                st.success("分析完成！")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("分析後影片")
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    else:
                        st.error("影片輸出失敗或檔案為空")

                with col2:
                    st.subheader("最大差值畫面")
                    if os.path.exists(max_frame_path):
                        st.image(max_frame_path, caption="最大肩膀/髖部差值畫面")
                    else:
                        st.write("未成功擷取最大差值畫面")

                st.subheader("總分")
                st.metric("揮桿評分", f"{result['score']} 分")

                st.subheader("基本數據")
                c1, c2, c3 = st.columns(3)
                c1.metric("最大肩膀角度", f"{result['max_shoulder_turn']:.2f}")
                c2.metric("最大髖部角度", f"{result['max_hip_turn']:.2f}")
                c3.metric("重心位移", f"{result['center_shift']:.2f}")

                c4, c5, c6 = st.columns(3)
                c4.metric("收桿肩膀角度", f"{result['finish_shoulder_angle']:.2f}")
                c5.metric("收桿手部高度", f"{result['finish_hand_y']:.2f}")
                c6.metric("肩膀高度", f"{result['shoulder_y']:.2f}")

                st.subheader("分析結果與建議")
                for item in result["analysis_results"]:
                    st.markdown(f"### {item['problem']}")
                    st.write(item["advice"])

            if os.path.exists(input_path):
                os.remove(input_path)