from ultralytics import YOLO
import cv2
import os
import serial
import time
import librosa
import numpy as np
import tensorflow as tf

# Function to turn all lights red using Arduino
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

# Load the TensorFlow Lite audio model
def load_audio_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the audio file
def preprocess_audio(audio_data, target_length=44032):
    try:
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        elif len(audio_data) > target_length:
            audio_data = audio_data[:target_length]

        return audio_data

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

# Predict whether siren is present in audio data
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

# File paths and settings
arduino_port = 'COM6'
arduino_baudrate = 9600
model_path = r"/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/dataset/converted_tflite/soundclassifier_with_metadata.tflite"
video_path = r"/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/in_footages/footage1.mp4"

# Define the output path for the processed video file
video_path_out = os.path.join('/Users/lakshan/Documents/Projects/I-CUBE/ambulance_detection_2/out_footages', 'output_video.mp4')

# Load the YOLO model for ambulance detection
model = YOLO('/opt/homebrew/runs/detect/train11/weights/last.pt')

# Load the TensorFlow Lite audio model
audio_model = load_audio_model(model_path)

# Video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Threshold for ambulance detection
threshold = 0.5

# Global variable for Arduino connection
arduino = None

# Process video frames
while ret:
    # Detect ambulances in the frame
    results = model(frame)[0]

    # Process detected boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw bounding box and label if score is above threshold
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Extract audio data from the frame
    audio_data = frame.mean(axis=1)  # Taking the mean of pixel values along the y-axis

    # Preprocess audio
    audio_data = preprocess_audio(audio_data)

    # Check if audio preprocessing was successful
    if audio_data is not None:
        # Make predictions for siren presence
        if predict_siren(audio_model, audio_data):
            print("Siren detected.")
            # Turn all traffic lights red if siren detected
            turn_all_lights_red()
        else:
            print("No siren detected.")
    else:
        print("Audio preprocessing failed.")

    # Write the processed frame to the output video
    out.write(frame)
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
