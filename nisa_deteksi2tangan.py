import cv2
import mediapipe as mp
import numpy as np

# === Inisialisasi MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Warna ===
BOX_COLOR = (0, 255, 0)        # Hijau untuk bounding box
TEXT_COLOR = (0, 255, 0)       # Hijau untuk label "Left/Right Hand"
LINE_COLOR = (0, 255, 0)       # Hijau untuk garis skeleton
POINT_COLOR = (0, 0, 255)      # Merah untuk titik sendi (keypoints)
TEXT_KP_COLOR = (0, 255, 255)  # Kuning cerah untuk angka keypoints

# === Fungsi menggambar tangan dengan keypoints dan label ===
def draw_hand_annotations(image, hand_landmarks, label):
    h, w, _ = image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # --- Bounding box hijau ---
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    cv2.rectangle(image, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), BOX_COLOR, 2)
    cv2.putText(image, label, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # --- Gambar skeleton (garis hijau + titik merah) ---
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=2),
        mp_drawing.DrawingSpec(color=POINT_COLOR, thickness=5, circle_radius=3)
    )

    # --- Gambar angka indeks di setiap titik sendi (warna kuning cerah) ---
    for idx, (x, y) in enumerate(coords):
        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_KP_COLOR, 2)

# === Mulai kamera ===
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=6,  # Bisa deteksi hingga 6 tangan
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca kamera.")
            continue

        # Flip untuk tampilan seperti mirror
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # === Jika ada tangan terdeteksi ===
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                score = results.multi_handedness[idx].classification[0].score
                label = f"{handedness} Hand ({score:.2f})"
                draw_hand_annotations(frame, hand_landmarks, label)

        # === Tampilkan hasil ===
        cv2.imshow("Multi-Hand Detection (Keypoints Merah & Angka Kuning)", frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()