import cv2
import mediapipe as mp
import numpy as np
import torch
from model_utils import ActionLSTM, calculate_angle

# --- 配置与初始化 ---
ACTIONS = ['Squat', 'Jumping Jack', 'Idle']
SEQ_LEN = 30
model = ActionLSTM(num_classes=len(ACTIONS))
# model.load_state_dict(torch.load('your_model.pth')) # 如有权重请取消注释
model.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def get_agent_advice(action, landmarks):
    """
    逻辑层 Agent：结合识别到的动作类型与物理角度给出具体反馈
    """
    advice = "观察中..."

    if action == 'Squat':
        # 获取左侧 髋-膝-踝 坐标
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)

        if angle < 95:
            advice = "非常好，下蹲深度标准！"
        elif angle > 140:
            advice = "下蹲深度不足，请尝试臀部继续下压。"
        else:
            advice = "深蹲进行中，注意背部挺直。"

    elif action == 'Jumping Jack':
        advice = "保持呼吸节奏，双臂尽量向头顶伸展。"

    return advice


# --- 视频流处理 ---
cap = cv2.VideoCapture(0)
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 提取132维特征向量
        cur_lm = []
        for lm in results.pose_landmarks.landmark:
            cur_lm.extend([lm.x, lm.y, lm.z, lm.visibility])

        sequence.append(cur_lm)
        sequence = sequence[-SEQ_LEN:]

        if len(sequence) == SEQ_LEN:
            input_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)
                idx = torch.argmax(prob).item()
                conf = prob[0][idx].item()

                curr_action = ACTIONS[idx]
                advice = get_agent_advice(curr_action, results.pose_landmarks.landmark)

                # 可视化输出
                cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, f"{curr_action} ({conf:.2f}) | Advice: {advice}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Action Recognition Agent', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()