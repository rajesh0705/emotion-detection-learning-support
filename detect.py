import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# ✅ Load Model Safely
# =========================
try:
    model = load_model("model/emotion_model.hdf5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# =========================
# ✅ Load Haarcascade (FIXED)
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    print("❌ Error loading Haarcascade")
    exit()

# =========================
# ✅ Start Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# =========================
# ✅ Full Screen Window
# =========================
cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Emotion Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🎥 Press ESC to exit")

# =========================
# 🔁 Main Loop
# =========================
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Failed to grab frame")
        break

    # Optional: resize for better display
    frame = cv2.resize(frame, (1280, 720))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        try:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face / 255.0
            face = np.reshape(face, (1,48,48,1))

            prediction = model.predict(face, verbose=0)
            label = emotion_labels[np.argmax(prediction)]

            # Draw rectangle
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # Put emotion text
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        except Exception as e:
            print("⚠️ Face error:", e)
            continue

    # Show output
    cv2.imshow("Emotion Detector", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# ✅ Release Resources
# =========================
cap.release()
cv2.destroyAllWindows()