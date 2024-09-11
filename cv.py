import cv2
import mediapipe as mp
import numpy as np
import time

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 設定眼睛閉合和嘴巴開合的閾值
EYE_AR_THRESH = 0.18  # 降低閾值以提高敏感度
YAWN_THRESH = 2  # 嘴巴開合比例閾值

# 計時器相關變量
EYE_CLOSED_TIME = 0  # 計算眼睛閉合的時間
YAWN_TIME = 0  # 計算嘴巴開合的時間
DELAY_SECONDS = 1.5  # 延遲3秒觸發

# 計數器
TOTAL_SUM = 0  # 記錄眼睛閉合和打哈欠的總和
EYE_CLOSED = False  # 眼睛閉合狀態
YAWNING = False  # 嘴巴打哈欠狀態

# 定義眼睛和嘴巴特徵點的索引
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17]  # 上唇中心, 下唇中心, 左嘴角, 右嘴角

def eye_aspect_ratio(landmarks, eye):
    points = [landmarks[i] for i in eye]
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def mouth_aspect_ratio(landmarks, mouth):
    points = [landmarks[i] for i in mouth]
    vertical = np.linalg.norm(points[1] - points[0])
    horizontal = np.linalg.norm(points[2] - points[3])
    mar = vertical / horizontal
    return mar

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("忽略空幀。")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x * image.shape[1], lm.y * image.shape[0], lm.z) for lm in face_landmarks.landmark])
            
            # 計算眼睛長寬比
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # 計算嘴巴長寬比
            mar = mouth_aspect_ratio(landmarks, MOUTH)

            # 檢查眼睛是否閉合
            if ear < EYE_AR_THRESH:
                EYE_CLOSED_TIME += 1  # 增加閉合時間
                if EYE_CLOSED_TIME >= DELAY_SECONDS * 30:  # 假設每秒30幀
                    if not EYE_CLOSED:
                        EYE_CLOSED = True
                        TOTAL_SUM += 1  # 觸發一次
            else:
                EYE_CLOSED_TIME = 0  # 重置計時器
                EYE_CLOSED = False

            # 檢查是否在打哈欠
            if mar < YAWN_THRESH:
                YAWN_TIME += 1  # 增加打哈欠時間
                if YAWN_TIME >= DELAY_SECONDS * 30:  # 假設每秒30幀
                    if not YAWNING:
                        YAWNING = True
                        TOTAL_SUM += 1  # 觸發一次
            else:
                YAWN_TIME = 0  # 重置計時器
                YAWNING = False

            # 在圖像上畫出眼睛和嘴巴
            for eye in [LEFT_EYE, RIGHT_EYE]:
                pts = landmarks[eye][:, :2].astype(int)
                cv2.polylines(image, [pts], True, (0, 255, 0), 1)

            mouth_pts = landmarks[MOUTH][:, :2].astype(int)
            cv2.polylines(image, [mouth_pts], True, (0, 255, 255), 1)

            # 顯示 EAR 和 MAR 值
            cv2.putText(image, f"EAR: {ear:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Total Sum: {TOTAL_SUM}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if TOTAL_SUM >= 3:
                cv2.putText(image, "Hey bro go to sleep", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()