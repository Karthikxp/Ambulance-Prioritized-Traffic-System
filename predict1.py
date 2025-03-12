import cv2
import numpy as np
import librosa
import tensorflow as tf
from ultralytics import YOLO


# Function to load the YOLO model for ambulance detection
def load_ambulance_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


# Function to detect ambulance in video frames
def detect_ambulance(model, frame, threshold=0.5):
    try:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    except Exception as e:
        print(f"Error detecting ambulance: {e}")


# Function to load the TensorFlow Lite model for siren detection
def load_siren_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TensorFlow Lite model: {e}")
        return None


# Function to preprocess audio file for siren detection
def preprocess_audio(file_path, target_length=44032):
    try:
        y, sr = librosa.load(file_path, sr=None)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        elif len(y) > target_length:
            y = y[:target_length]

        return y, sr
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None


# Function to predict siren presence
def predict_siren(interpreter, audio_data):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(audio_data, axis=0))
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        is_siren = predictions[0][0] > 0.5
        return is_siren
    except Exception as e:
        print(f"Error predicting siren: {e}")
        return False


# Load YOLO model for ambulance detection
model_path_ambulance = '/opt/homebrew/runs/detect/train11/weights/last.pt'
ambulance_model = load_ambulance_model(model_path_ambulance)

# Load TensorFlow Lite model for siren detection
model_path_siren = r"/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/dataset/converted_tflite/soundclassifier_with_metadata.tflite"
siren_model = load_siren_model(model_path_siren)

# Paths to video and audio files
video_path = '/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/in_footages/footage1.mp4'
audio_file_path = '/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/in_footages/audio.mp3'

# Capture video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Read audio file
audio_data, sample_rate = preprocess_audio(audio_file_path)

# Detection loop
while ret:
    # Detect ambulance in video frame
    detect_ambulance(ambulance_model, frame)

    # Check if preprocessing was successful
    if audio_data is not None and sample_rate is not None:
        # Predict siren presence
        if predict_siren(siren_model, audio_data):
            print("Siren detected.")
        else:
            print("No siren detected.")
    else:
        print("Audio preprocessing failed.")

    # Display frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
