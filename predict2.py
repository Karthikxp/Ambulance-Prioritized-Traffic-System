import cv2
import numpy as np
import librosa
import tensorflow as tf
import serial
import time
from ultralytics import YOLO

# Arduino setup
arduino_port = 'COM6'
arduino_baudrate = 9600
arduino = None

def load_audio_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_audio(file_path, target_length=44032):
    try:
        y, sr = librosa.load(file_path)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        elif len(y) > target_length:
            y = y[:target_length]

        return y, sr

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None

def predict_ambulance(interpreter, audio_data):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(audio_data, axis=0))
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])

        is_ambulance = predictions[0][0] > 0.5
        return is_ambulance

    except Exception as e:
        print(f"Error predicting ambulance: {e}")
        return False

def turn_all_lights_red():
    global arduino
    try:
        if arduino is None:
            arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
            time.sleep(2)
        
        # Ensure the connection is open
        if arduino.is_open:
            arduino.write(b'R')
            print("R is sent to board")
        else:
            print("Arduino connection is not open.")

    except Exception as e:
        print(f"Error sending 'R' to Arduino: {e}")

# Load the YOLO model for ambulance detection
model_path_ambulance = '/opt/homebrew/runs/detect/train11/weights/last.pt'
ambulance_model = YOLO(model_path_ambulance)

# Load the TensorFlow Lite model for siren detection
model_path_siren = r"/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/dataset/converted_tflite/soundclassifier_with_metadata.tflite"
siren_model = load_audio_model(model_path_siren)

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
    results = ambulance_model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:  # Adjust threshold as needed
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Check if preprocessing was successful
    if audio_data is not None and sample_rate is not None:
        # Predict siren presence
        if predict_ambulance(siren_model, audio_data):
            print("Siren detected.")
            # Turn all traffic lights red
            turn_all_lights_red()
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
