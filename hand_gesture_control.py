import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

filter_length = 5
x_positions = deque(maxlen=filter_length)
y_positions = deque(maxlen=filter_length)

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Acessou a câmera com índice {i}")
        break
    else:
        print(f"Índice {i} não funcionou.")
        cap.release()

if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1])
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])

            x_positions.append(x)
            y_positions.append(y)

            avg_x = int(sum(x_positions) / len(x_positions))
            avg_y = int(sum(y_positions) / len(y_positions))

            cv2.circle(image, (avg_x, avg_y), 10, (0, 255, 0), -1)

            pyautogui.moveTo(avg_x, avg_y)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            distance = calculate_distance(index_finger_tip, thumb_tip)

            if distance < 0.05:
                pyautogui.click()

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
