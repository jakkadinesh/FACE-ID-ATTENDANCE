import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
from datetime import datetime
import pandas as pd
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR, MODEL_DIR,
    FACE_DETECTION_THRESHOLD, RECOGNITION_THRESHOLD, IMAGE_SIZE
)

class FaceRecognizer:
    def __init__(self):
        # Load models
        self.mtcnn = MTCNN(keep_all=True, thresholds=[FACE_DETECTION_THRESHOLD]*3, device='cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Load classifier and label encoder
        with open(MODEL_DIR / 'classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(MODEL_DIR / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Attendance tracking
        self.attendance_log = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
        self.recognized_students = set()
        
        # Initialize attendance file if it doesn't exist
        if not self.attendance_log.exists():
            pd.DataFrame(columns=['student_id', 'timestamp']).to_csv(self.attendance_log, index=False)

    def recognize_faces(self, frame):
        """Recognize faces in a frame and mark attendance."""
        # Convert to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(pil_img)
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < FACE_DETECTION_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Extract face
                face = pil_img.crop((x1, y1, x2, y2))
                face = face.resize((IMAGE_SIZE, IMAGE_SIZE))
                face_tensor = torch.FloatTensor(np.array(face).transpose(2, 0, 1)) / 255.0
                
                # Get embedding
                embedding = self.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                
                # Predict
                predictions = self.classifier.predict_proba(embedding)
                max_prob = np.max(predictions)
                
                if max_prob > RECOGNITION_THRESHOLD:
                    pred_id = np.argmax(predictions)
                    student_id = self.label_encoder.inverse_transform([pred_id])[0]
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_id} ({max_prob:.2f})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Mark attendance if not already recognized
                    if student_id not in self.recognized_students:
                        self._mark_attendance(student_id)
                        self.recognized_students.add(student_id)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return frame

    def _mark_attendance(self, student_id):
        """Record attendance in the log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_record = pd.DataFrame([[student_id, timestamp]], columns=['student_id', 'timestamp'])
        
        # Append to existing file
        new_record.to_csv(self.attendance_log, mode='a', header=False, index=False)
        print(f"Attendance marked for {student_id} at {timestamp}")

def live_recognition():
    """Run live face recognition from webcam."""
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    print("Starting live recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Recognize faces
        frame = recognizer.recognize_faces(frame)
        
        # Display
        cv2.imshow('Face Attendance System', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_recognition()