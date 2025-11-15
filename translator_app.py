# Importing Libraries
import time
import pickle
import statistics
import random
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configurations
MODEL_PATH = "sign_seq_translator_v2.h5"
LABEL_MAP_PATH = "label_map.pkl"
DATASET_DIR = "dataset_sequences"
SEQUENCE_LENGTH = 30
N_FEATURES = 60
PRED_HISTORY = 8
CONFIDENCE_THRESHOLD = 0.65
SENTENCE_TIMEOUT = 3.5 # Seconds of pause before clearing the translation bar


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_results(results, required_features=N_FEATURES):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    
    if len(keypoints) < required_features:
        keypoints.extend([0.0] * (required_features - len(keypoints)))
    elif len(keypoints) > required_features:
        keypoints = keypoints[:required_features]
    return np.array(keypoints, dtype=np.float32)

"""CLASS 1: DataRecorder:
Encapsulates all logic for collecting and saving training data."""
class DataRecorder:    
    def __init__(self, dataset_dir=DATASET_DIR, seq_len=SEQUENCE_LENGTH, n_features=N_FEATURES):
        self.dataset_dir = Path(dataset_dir)
        self.seq_len = seq_len
        self.n_features = n_features
        self.current_label = None
        self.recording = False
        self._buffer = []

    def set_label(self, label: str):
        self.current_label = label
        (self.dataset_dir / label).mkdir(parents=True, exist_ok=True)
        print(f"[RECORDER] Current label set to: {label}")

    def start(self):
        if self.current_label is None:
            print("[RECORDER] Set a label before recording.")
            return
        self.recording = True
        self._buffer = []
        print(f"[RECORDER] Recording started for '{self.current_label}'...")

    def stop_and_save(self):
        if not self._buffer:
            print("[RECORDER] No frames recorded, nothing saved.")
            self.recording = False
            return
            
        seq = np.array(self._buffer, dtype=np.float32)
        
        if seq.shape[0] < self.seq_len:
            pad_len = self.seq_len - seq.shape[0]
            seq = np.vstack([seq, np.zeros((pad_len, self.n_features), dtype=np.float32)])
        elif seq.shape[0] > self.seq_len:
            seq = seq[:self.seq_len, :]
            
        ts = int(time.time() * 1000)
        next_i = random.randint(100, 999) 
        filename = self.dataset_dir / self.current_label / f"seq_{ts}_{next_i}.pkl"
        
        with open(filename, "wb") as f:
            pickle.dump(seq, f)
        print(f"[RECORDER] Saved sequence: {filename} ({seq.shape[0]} frames)")
        self.recording = False
        self._buffer = []

    def add_frame(self, features):
        if self.recording and features.shape[0] == self.n_features:
            self._buffer.append(features)

"""CLASS 2: SignRecognizer:
Encapsulates model loading, feature buffering, and prediction logic."""
class SignRecognizer:
    def __init__(self, model_path: str, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len = seq_len
        self.model_path = model_path
        self.ml_model = None
        self.id2label = {}
        self.n_features = N_FEATURES
        self.sequence_buffer = deque(maxlen=seq_len)
        self.pred_history = deque(maxlen=PRED_HISTORY)
        self._load_labels()
        self.load_model(model_path) 
        
        if self.ml_model:
            try:
                self.n_features = self.ml_model.input_shape[-1]
            except Exception:
                 pass

    def _load_labels(self):
        if Path(LABEL_MAP_PATH).exists():
            with open(LABEL_MAP_PATH, "rb") as f:
                label2idx = pickle.load(f)
            self.id2label = {v: k for k, v in label2idx.items()}
            print(f"[MODEL] Loaded {len(self.id2label)} labels from {LABEL_MAP_PATH}.")
        else:
            self.id2label = {0: "_DEFAULT_", 5: "_HOLD_"}
            print("[MODEL] label_map.pkl not found. Ready for recording.")

    def load_model(self, model_path: str):
        try:
            tf.get_logger().setLevel('ERROR') 
            if model_path and Path(model_path).exists():
                self.ml_model = tf.keras.models.load_model(model_path)
                print(f"[MODEL] Model loaded successfully from {model_path}.")
            else:
                raise FileNotFoundError("Model file not found.")
        except Exception as e:
            print(f"WARNING: Model load failed. Waiting for training. Error: {e}")
            self.ml_model = None
            
    def add_keypoints(self, features: np.ndarray):
        self.sequence_buffer.append(features)

    def predict(self) -> tuple:
        if len(self.sequence_buffer) < self.seq_len or self.ml_model is None:
            return 5, 0.0 # _HOLD_ ID (fallback)
        
        window = np.array(list(self.sequence_buffer), dtype=np.float32)
        input_data = np.expand_dims(window, axis=0)
        
        predictions = self.ml_model.predict(input_data, verbose=0)[0]
        pred_id = int(np.argmax(predictions))
        conf = float(predictions[pred_id])
        
        if conf < CONFIDENCE_THRESHOLD:
             return 5, conf # _HOLD_ ID

        self.pred_history.append(pred_id)
        try:
            stable_id = statistics.mode(self.pred_history)
        except statistics.StatisticsError:
            stable_id = self.pred_history[-1]

        return stable_id, conf

"""CLASS 3: TextTranslator:
Encapsulates the linguistic logic of assembling confirmed signs into a sentence"""
class TextTranslator:    
    def __init__(self, id2label: dict, min_hold_time=0.6, sentence_timeout=SENTENCE_TIMEOUT):
        self.id2label = id2label
        self.min_hold_time = min_hold_time
        self.sentence_timeout = sentence_timeout
        self.last_seen_id = None
        self.last_seen_time = time.time()
        self.last_confirmed = None
        self.current_sentence = ""
        self.last_confirmed_time = time.time()

    def update(self, sign_id: int, confidence: float) -> str:
        t = time.time()
        
        # Check for sentence timeout FIRST..
        # If a sentence exists AND we haven't confirmed a new word in 3.5 seconds...
        if self.current_sentence and (t - self.last_confirmed_time > self.sentence_timeout):
            print(f"[TRANSLATOR DEBUG]: Sentence timeout ({self.sentence_timeout}s). Clearing bar.")
            self.current_sentence = ""
            self.last_confirmed = None

        if sign_id == 5 or confidence < CONFIDENCE_THRESHOLD:
            self.last_seen_id = None
            return self.current_sentence.strip()

        if sign_id != self.last_seen_id:
            self.last_seen_id = sign_id
            self.last_seen_time = t
            return self.current_sentence.strip()

        if t - self.last_seen_time >= self.min_hold_time:
            label = self.id2label.get(sign_id, "[UNK]")
            
            if self.last_confirmed != sign_id:
                if label == "_SPACE_":
                    if self.current_sentence.strip() and not self.current_sentence.endswith(" "):
                        self.current_sentence += " "
                elif label not in ("[UNK]", "_HOLD_"):
                    self.current_sentence += label + " "
                    
                self.last_confirmed = sign_id
                self.last_confirmed_time = t # Reset timeout timer on new word
                print(f"[TRANSLATOR DEBUG]: Confirmed sign: {label}")
                
            self.last_seen_id = None
            self.last_seen_time = t

        return self.current_sentence.strip()
    
"""CLASS 4: SignLanguageApp (Composition & Orchestration):
The main application class. It composes all other OOPS components to run the app."""
class SignLanguageApp:
    def __init__(self, model_path=MODEL_PATH):
        print("Initializing Sign Language Translator Application...")
        
        self.hands_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.camera = cv2.VideoCapture(0)
        
        # OOPS Composition: Create instances of other classes
        self.recognizer = SignRecognizer(model_path=model_path, seq_len=SEQUENCE_LENGTH)
        self.recorder = DataRecorder(seq_len=SEQUENCE_LENGTH, n_features=self.recognizer.n_features)
        self.assembler = TextTranslator(id2label=self.recognizer.id2label)
        
        self.running = True
        self.app_state = "WELCOME" # App state management
        self.mode = "INFER"
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            "bg": (40, 30, 30),
            "white": (255, 255, 255),
            "light_gray": (200, 200, 200),
            "blue": (255, 150, 0),
            "yellow": (0, 220, 255),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "shadow": (0, 0, 0)
        }
        self.window_name = "Sign Language Translator - OOPS Project"

    def _create_welcome_frame(self):
        """Creates the static welcome screen frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = self.colors["bg"]
        
        def add_text(text, origin, scale, color, thickness):
            cv2.putText(frame, text, origin, self.font, scale, color, thickness, cv2.LINE_AA)

        add_text("Sign Language Translator", (280, 120), 2.0, self.colors["white"], 3)
        add_text("An initiative to bring sign language to everyone.", (330, 170), 0.9, self.colors["light_gray"], 2)
        add_text("CONTROLS:", (100, 280), 1.0, self.colors["blue"], 2)
        add_text("R  - Toggle RECORD / INFERENCE mode", (120, 340), 0.9, self.colors["white"], 2)
        add_text("L  - Set Label (in Record mode, check console)", (120, 390), 0.9, self.colors["white"], 2)
        add_text("Space  - Start / Stop Recording (in Record mode)", (120, 440), 0.9, self.colors["white"], 2)
        add_text("Q  - Quit Application", (120, 490), 0.9, self.colors["white"], 2)
        add_text("Press [ENTER] to Start", (420, 650), 1.0, self.colors["yellow"], 2)
        
        return frame

    def run(self):
        if not self.camera.isOpened():
             print("ERROR: Cannot open camera.")
             self.running = False
        
        while self.running:
            key = cv2.waitKey(1) & 0xFF

            if self.app_state == "WELCOME":
                frame = self._create_welcome_frame()
                if key == 13: # 13 is the 'Enter' key
                    self.app_state = "RUNNING"
                    print("Starting application...")
            
            elif self.app_state == "RUNNING":
                ret, frame = self.camera.read()
                if not ret: continue
                
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_model.process(img_rgb)

                features = extract_keypoints_from_results(results, required_features=self.recognizer.n_features)
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                
                self.recorder.add_frame(features) 
                self.recognizer.add_keypoints(features)

                sign_id, conf = self.recognizer.predict()
                sentence = self.assembler.update(sign_id, conf) 
                
                annotated_frame = self._visualize(frame, sign_id, conf, sentence)
                frame = annotated_frame

                if key == ord('q'): self.running = False
                elif key == ord('r'): self._toggle_mode()
                elif key == ord('l') and self.mode == "RECORD": self._set_label()
                elif key == 32 and self.mode == "RECORD":
                    if not self.recorder.recording: self.recorder.start()
                    else: self.recorder.stop_and_save()

            cv2.imshow(self.window_name, frame)

        self.camera.release()
        cv2.destroyAllWindows()

    def _visualize(self, frame, sign_id, confidence, sentence):
        """Drawing the UI elements on the frame."""
        height, width, _ = frame.shape
        raw_label = self.recognizer.id2label.get(sign_id, "WAITING...")
        
        if raw_label in ["_SPACE_", "_HOLD_"]:
            display_label = "Processing..." if confidence >= CONFIDENCE_THRESHOLD else "WAITING..."
        else:
            display_label = raw_label
        
        overlay = frame.copy()
        alpha = 0.6
        
        cv2.rectangle(overlay, (0, height - 70), (width, height), self.colors["bg"], -1)
        cv2.rectangle(overlay, (10, 10), (450, 70), self.colors["bg"], -1)
        
        mode_color = self.colors["red"] if self.mode == "RECORD" else self.colors["green"]
        cv2.rectangle(overlay, (width - 250, 10), (width - 10, 70), mode_color, -1)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "TRANSLATION:", (20, height - 45), self.font, 0.7, self.colors["light_gray"], 2)
        cv2.putText(frame, sentence, (200, height - 45), self.font, 0.8, self.colors["yellow"], 2)
        
        pred_color = self.colors["green"] if confidence >= CONFIDENCE_THRESHOLD else self.colors["light_gray"]
        cv2.putText(frame, "PREDICTION:", (25, 45), self.font, 0.7, self.colors["light_gray"], 2)
        cv2.putText(frame, f"{display_label} ({confidence*100:.0f}%)", (170, 45), self.font, 0.8, pred_color, 2)
        
        mode_text = f"MODE: {self.mode}"
        if self.mode == "RECORD" and self.recorder.recording:
            mode_text = f"REC: {self.recorder.current_label}"
            
        cv2.putText(frame, mode_text, (width - 235, 45), self.font, 0.8, self.colors["white"], 2)

        return frame

    def _toggle_mode(self):
        if self.mode == "INFER":
            self.mode = "RECORD"
            print("[MODE] Switched to RECORD. Press 'l' to set label. 'Space' to start/stop recording.")
        else:
            if self.recorder.recording:
                 self.recorder.stop_and_save()
            self.mode = "INFER"
            print("[MODE] Switched to INFERENCE.")

    def _set_label(self):
        try:
            label = input("Enter label name (e.g., hi, thank you): ").strip()
            if label:
                self.recorder.set_label(label)
        except EOFError:
            print("Label input cancelled.")

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()