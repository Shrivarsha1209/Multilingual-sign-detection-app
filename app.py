from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
from ultralytics import YOLO
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import copy
import itertools
import os
from huggingface_hub import hf_hub_download
import mediapipe as mp

app = Flask(__name__)

# Paths for models
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ASL_MODEL_REPO_ID = "atalaydenknalbant/asl-yolo-models"  # Replace with repo ID containing yolov11s.pt
ASL_MODEL_FILENAME = "yolo11s.pt"  # YOLO model filename

# ISL model path
isl_model_path = os.path.join(os.getcwd(), "models", "model (2).h5")

# Load ASL YOLO model
print("Downloading ASL YOLO model from Hugging Face...")
asl_model_path = hf_hub_download(repo_id=ASL_MODEL_REPO_ID, filename=ASL_MODEL_FILENAME)
asl_model = YOLO(asl_model_path)

# Load ISL TensorFlow model
print("Loading ISL TensorFlow model...")
isl_model = keras.models.load_model(isl_model_path)

# MediaPipe setup for ISL
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Alphabet for ISL
isl_alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)


def annotate_asl(image_bgr):
    """Run ASL YOLO model and annotate image."""
    results = asl_model.predict(image_bgr)
    return results[0].plot()


def annotate_isl(image_bgr):
    """Run ISL model with MediaPipe and TensorFlow for predictions."""
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image_bgr, hand_landmarks)
                pre_processed_landmarks = pre_process_landmark(landmark_list)
                df = pd.DataFrame(pre_processed_landmarks).transpose()

                # Predict using the TensorFlow model
                predictions = isl_model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                label = isl_alphabet[predicted_class]

                # Draw landmarks and label
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                cv2.putText(
                    image_bgr, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2
                )

        return image_bgr


def calc_landmark_list(image, landmarks):
    """Calculate hand landmarks."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    """Normalize landmarks."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return [n / max_value for n in temp_landmark_list]


@app.route("/")
def home():
    """Render the homepage with language selection."""
    return render_template("index.html")


@app.route("/set_language", methods=["POST"])
def set_language():
    """Set the selected language."""
    language = request.form.get("language")
    return redirect(url_for("webcam", language=language))


@app.route("/webcam/<language>")
def webcam(language):
    """Render the webcam interface."""
    return render_template("webcam.html", language=language)


def generate_frames(language):
    """Stream webcam feed with ASL or ISL detection."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if language == "asl":
            annotated_frame = annotate_asl(frame)
        elif language == "isl":
            annotated_frame = annotate_isl(frame)
        else:
            cap.release()
            return "Invalid language selected."

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


@app.route("/video_feed/<language>")
def video_feed(language):
    """Stream video feed for the selected language."""
    return Response(generate_frames(language), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
