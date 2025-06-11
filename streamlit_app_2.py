import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from datetime import datetime
from torchvision import transforms

# Configuration
DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("./models")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ATTENDANCE_DIR = DATA_DIR / "attendance"

for dir_path in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(
        image_size=160,
        margin=20,  # 20% margin as required
        min_face_size=60,  # Increased from 40 to reduce small face false positives
        thresholds=[0.9, 0.95, 0.95],  # More strict thresholds
        keep_all=False,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device

def preprocess_frame(frame):
    """Enhance image quality for better face detection"""
    # Enhance contrast and brightness
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    
    # Apply slight sharpening for better facial features
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

def update_confidence(buffer, new_value, alpha=0.3):
    """Implement exponential weighted average for confidence scores"""
    if not buffer:
        return [new_value]
    weighted_avg = alpha * new_value + (1 - alpha) * buffer[-1]
    buffer.append(weighted_avg)
    if len(buffer) > 5:  # Keep buffer size at 5
        buffer.pop(0)
    return buffer

def main():
    st.set_page_config(page_title="Face Attendance System", layout="wide")
    st.title("Face-Based Attendance System")

    menu = ["Capture Faces", "Train Model", "Recognize Faces", "View Attendance"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Capture Faces":
        st.header("Capture New Face Samples")
        student_id = st.text_input("Enter Student ID:")
        student_name = st.text_input("Enter Student Name (Optional):")
        num_samples = st.slider("Number of Samples to Capture", 10, 100, 20)
        
        col1, col2 = st.columns(2)
        with col1:
            capture_btn = st.button("Start Capture")
        with col2:
            preview = st.checkbox("Enable Preview", value=True)

        if capture_btn:
            if not student_id:
                st.warning("Please enter a Student ID")
            else:
                capture_faces(student_id, num_samples, preview, student_name)

    elif choice == "Train Model":
        st.header("Train Face Recognition Model")
        
        col1, col2 = st.columns(2)
        with col1:
            augmentation = st.checkbox("Use Data Augmentation", value=True)
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                embeddings, labels = process_faces(augmentation)
                train_classifier(embeddings, labels, cv_folds)
            st.success("Model trained successfully!")

    elif choice == "Recognize Faces":
        st.header("Live Face Recognition")
        
        col1, col2 = st.columns(2)
        with col1:
            recognition_threshold = st.slider("Recognition Confidence Threshold", 0.5, 1.0, 0.7)
        with col2:
            frame_skip = st.slider("Process every N frames", 1, 5, 2, help="Higher values improve performance but may reduce responsiveness")
        
        if st.button("Start Recognition"):
            recognize_faces(recognition_threshold, frame_skip)

    elif choice == "View Attendance":
        st.header("Attendance Records")
        
        attendance_files = sorted(ATTENDANCE_DIR.glob("*.csv"), reverse=True)

        if not attendance_files:
            st.info("No attendance records found yet.")
        else:
            selected_file = st.selectbox("Select a date", [f.name for f in attendance_files])
            df = pd.read_csv(ATTENDANCE_DIR / selected_file)
            
            # Add filtering options
            if 'Student ID' in df.columns:
                student_filter = st.multiselect("Filter by Student ID", options=sorted(df['Student ID'].unique()))
                if student_filter:
                    df = df[df['Student ID'].isin(student_filter)]
            
            # Add time range filtering
            if 'Timestamp' in df.columns:
                df['Time'] = pd.to_datetime(df['Timestamp']).dt.time
                start_time = st.slider("Start Time", 
                                     min_value=datetime.strptime("00:00", "%H:%M").time(),
                                     max_value=datetime.strptime("23:59", "%H:%M").time(),
                                     value=datetime.strptime("00:00", "%H:%M").time(),
                                     format="HH:mm")
                end_time = st.slider("End Time", 
                                   min_value=datetime.strptime("00:00", "%H:%M").time(),
                                   max_value=datetime.strptime("23:59", "%H:%M").time(),
                                   value=datetime.strptime("23:59", "%H:%M").time(),
                                   format="HH:mm")
                
                df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
            
            # Display filtered data
            st.dataframe(df)
            
            # Export option
            if st.button("Export to CSV"):
                export_path = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(export_path, index=False)
                st.success(f"Exported to {export_path}")


def capture_faces(student_id, num_samples, preview=True, student_name=None):
    mtcnn, resnet, device = load_models()
    student_dir = RAW_DIR / student_id
    student_dir.mkdir(exist_ok=True)
    
    # Save student metadata if name is provided
    if student_name:
        metadata = {"id": student_id, "name": student_name, "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        with open(student_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
    
    # Improve camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)

    st.warning("Capturing samples automatically...")
    image_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    count = 0
    total_frames = 0
    quality_threshold = 0.92  # Face detection quality threshold

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        total_frames += 1
        
        # Enhance frame quality
        processed_frame = preprocess_frame(frame)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        try:
            boxes, probs = mtcnn.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                best_idx = np.argmax(probs)
                box = boxes[best_idx]
                prob = probs[best_idx]
                
                if prob < quality_threshold:
                    status_text.warning(f"Low quality face detected ({prob:.2f}). Please adjust position/lighting.")
                    if preview:
                        # Draw yellow box for low quality face
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"Quality: {prob:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        image_placeholder.image(frame, channels="BGR", use_container_width=True)
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Add margin around face (20% as required)
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(frame.shape[1], x2 + margin_x)
                y2 = min(frame.shape[0], y2 + margin_y)
                
                face_crop = processed_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Only save every 5th good frame to ensure diversity
                if total_frames % 5 == 0:
                    img_path = student_dir / f"{student_id}_{count:04d}.jpg"
                    cv2.imwrite(str(img_path), face_crop)
                    count += 1
                    progress_bar.progress(count / num_samples)

                if preview:
                    # Draw green rectangle around the face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Captured {count}/{num_samples}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    image_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            continue
            
        # Small delay to prevent too rapid captures
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    st.success(f"Completed capturing {count} samples for {student_id}")
    
    # Show some of the captured samples
    st.subheader("Sample Images Captured")
    col1, col2, col3 = st.columns(3)
    
    sample_images = list(student_dir.glob('*.jpg'))
    if len(sample_images) > 0:
        with col1:
            if len(sample_images) > 0:
                st.image(str(sample_images[0]))
        with col2:
            if len(sample_images) > len(sample_images)//2:
                st.image(str(sample_images[len(sample_images)//2]))
        with col3:
            if len(sample_images) > len(sample_images)-1:
                st.image(str(sample_images[-1]))


def process_faces(use_augmentation=True):
    mtcnn, resnet, device = load_models()
    embeddings = []
    labels = []

    student_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data augmentation transforms
    if use_augmentation:
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        aug_transforms = [
            # Original
            basic_transform,
            # Horizontal flip
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            # Brightness variation
            transforms.Compose([
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            # Contrast variation
            transforms.Compose([
                transforms.ColorJitter(contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ]
    else:
        # Just basic normalization if no augmentation
        aug_transforms = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])]

    for i, student_dir in enumerate(student_dirs):
        student_id = student_dir.name
        image_files = list(student_dir.glob('*.jpg'))
        processed_dir = PROCESSED_DIR / student_id
        processed_dir.mkdir(exist_ok=True)

        status_text.text(f"Processing {student_id} ({i+1}/{len(student_dirs)})")
        
        successful_extractions = 0
        
        for img_path in image_files:
            try:
                # Load image
                img = Image.open(img_path)
                
                # Process with each transform if using augmentation
                for transform_idx, transform in enumerate(aug_transforms):
                    try:
                        # Apply augmentation or just use the original image
                        if transform_idx == 0:
                            # Try direct MTCNN detection first (original image)
                            face = mtcnn(img)
                        else:
                            # For augmented versions
                            img_tensor = transform(img).unsqueeze(0).to(device)
                            img_tensor = img_tensor * 0.5 + 0.5  # Denormalize for MTCNN
                            img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                            img_np = img_np.astype(np.uint8)
                            img_pil = Image.fromarray(img_np)
                            face = mtcnn(img_pil)
                        
                        if face is None:
                            continue
                    except Exception as e:
                        continue
                    
                    face = face.to(device)
                    
                    # Get embedding
                    with torch.no_grad():
                        embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()
                    embeddings.append(embedding.flatten())
                    labels.append(student_id)
                    
                    # Only save processed image for the original (non-augmented) version
                    if transform_idx == 0:
                        processed_path = processed_dir / f"{img_path.stem}_proc{img_path.suffix}"
                        face_np = face.permute(1, 2, 0).cpu().numpy()
                        face_np = (face_np * 255).astype(np.uint8)
                        cv2.imwrite(str(processed_path), cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                    
                    successful_extractions += 1
                
            except Exception as e:
                st.warning(f"Error processing {img_path}: {str(e)}")
                continue

        progress_bar.progress((i + 1) / len(student_dirs))
        status_text.text(f"Processed {successful_extractions} images for {student_id}")

    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # Save embeddings and labels
        np.save(EMBEDDINGS_DIR / 'embeddings.npy', embeddings)
        np.save(EMBEDDINGS_DIR / 'labels.npy', labels)
        
        st.success(f"Processed {len(embeddings)} face samples from {len(set(labels))} students")
        return embeddings, labels
    else:
        st.error("No valid face embeddings were extracted. Please check your image data.")
        return np.array([]), np.array([])

def train_classifier(embeddings, labels, cv_folds=5):
    if len(embeddings) == 0 or len(labels) == 0:
        st.error("No data available for training. Please capture faces first.")
        return
        
    st.info(f"Training with {len(embeddings)} samples across {len(set(labels))} classes")
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    normalizer = Normalizer(norm='l2')
    normalized_embeddings = normalizer.fit_transform(embeddings)
    
    # Use grid search with cross-validation for better SVM parameters
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    
    progress_text = st.empty()
    progress_text.text("Performing cross-validation to find optimal parameters...")
    
    clf = GridSearchCV(
        SVC(kernel='linear', probability=True, class_weight='balanced'),
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        verbose=1
    )
    
    clf.fit(normalized_embeddings, encoded_labels)
    
    st.info(f"Best parameters: C={clf.best_params_['C']}")
    st.info(f"Cross-validation accuracy: {clf.best_score_:.4f}")

    # Save the models and preprocessing components
    with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

    with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open(MODEL_DIR / 'normalizer.pkl', 'wb') as f:
        pickle.dump(normalizer, f)
        
    # Also save some metadata about the model
    model_meta = {
        "trained_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "num_samples": len(embeddings),
        "num_classes": len(set(labels)),
        "classes": list(set(labels)),
        "accuracy": clf.best_score_,
        "parameters": clf.best_params_
    }
    
    with open(MODEL_DIR / 'model_meta.pkl', 'wb') as f:
        pickle.dump(model_meta, f)

# Global dictionary to track attendance for deduplication
attendance_log = {}

def mark_attendance(student_id, confidence):
    """Enhanced attendance logging with confidence scores"""
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    filename = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    confidence_str = f"{confidence:.2f}"
    
    # Generate a session ID if it doesn't exist
    if 'current_session' not in st.session_state:
        st.session_state.current_session = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Check if student metadata exists to get the name
    student_name = "Unknown"
    metadata_path = RAW_DIR / student_id / "metadata.pkl"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                if 'name' in metadata:
                    student_name = metadata['name']
        except:
            # If error loading metadata, just use ID
            pass
    
    if not filename.exists():
        df = pd.DataFrame(columns=['Student ID', 'Name', 'Timestamp', 'Confidence', 'Session ID'])
    else:
        df = pd.read_csv(filename)
    
    # Only mark attendance if the student hasn't been seen in the last 10 minutes
    now = datetime.now()
    last_seen = attendance_log.get(student_id)
    if last_seen is None or (now - last_seen).seconds >= 600:  # 10 minutes
        new_row = pd.DataFrame([[student_id, student_name, timestamp, confidence_str, st.session_state.current_session]], 
                             columns=['Student ID', 'Name', 'Timestamp', 'Confidence', 'Session ID'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
        attendance_log[student_id] = now
        return True
    return False

def recognize_faces(threshold, frame_skip=2):
    mtcnn, resnet, device = load_models()

    try:
        with open(MODEL_DIR / 'classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open(MODEL_DIR / 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open(MODEL_DIR / 'normalizer.pkl', 'rb') as f:
            normalizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return

    # Settings for recognition
    BUFFER_SIZE = 5  # Number of frames to consider for smoothing
    CONFIDENCE_THRESHOLD = threshold  # Minimum confidence for marking attendance
    
    # Create prediction buffers for each detected face
    prediction_buffer = {}
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
    
    # Improve camera settings for better image quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)

    st.warning("Recognition in progress. Click 'Stop Recognition' to end.")
    image_placeholder = st.empty()
    status_text = st.empty()
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        fps_metric = st.empty()
    with metrics_col2:
        faces_metric = st.empty()
    
    stop_button_col = st.empty()
    stop_button = stop_button_col.button("Stop Recognition")
    
    # Track recognized students for attendance
    recognized_students = set()
    
    # Performance tracking
    frame_count = 0
    skip_count = 0
    start_time = time.time()
    fps = 0
    face_count = 0

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
            
        # Skip frames if needed for performance
        skip_count += 1
        if skip_count % frame_skip != 0:
            # Still display the frame but skip processing
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            image_placeholder.image(frame, channels="BGR", use_container_width=True)
            continue

        # Apply preprocessing to improve face detection
        processed_frame = preprocess_frame(frame)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Add FPS counter to display
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            detection_result = mtcnn.detect(frame_rgb)
            
            if len(detection_result) == 3:
                boxes, probs, landmarks = detection_result
            else:
                boxes, probs = detection_result
                landmarks = None
            
            face_count = 0 if boxes is None else len(boxes)
            faces_metric.metric("Faces Detected", face_count)
            
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    # Skip low quality detections
                    if prob < 0.92:
                        continue

                    face_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are valid
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 6400:  # Skip small faces (80x80 pixels)
                        continue
                        
                    # Add margin around the face for better recognition (20% as specified)
                    margin_x = int((x2 - x1) * 0.2)
                    margin_y = int((y2 - y1) * 0.2)
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(frame.shape[1], x2 + margin_x)
                    y2 = min(frame.shape[0], y2 + margin_y)
                    
                    # Extract face region
                    face_img = frame_rgb[y1:y2, x1:x2]
                    
                    # Convert to PIL
                    face_pil = Image.fromarray(face_img)
                    
                    try:
                        # Process through MTCNN
                        face_tensor = mtcnn(face_pil)
                        
                        if face_tensor is None:
                            continue
                            
                        # Process the face tensor
                        face_tensor = face_tensor.to(device)
                        
                        # Get embedding with normalization
                        with torch.no_grad():
                            embedding = resnet(face_tensor.unsqueeze(0)).cpu().numpy()
                        
                        normalized_embedding = normalizer.transform(embedding)
                        
                        # Get prediction probabilities
                        proba = clf.predict_proba(normalized_embedding)[0]
                        max_prob = np.max(proba)
                        pred_id = np.argmax(proba)
                        student_id = le.inverse_transform([pred_id])[0]
                        
                        # Create a unique face tracking ID for this frame
                        face_id = f"{student_id}_{i}"
                        
                        if face_id not in prediction_buffer:
                            prediction_buffer[face_id] = []
                        
                        # Update confidence with weighted average
                        prediction_buffer[face_id] = update_confidence(prediction_buffer[face_id], max_prob)
                            
                        # Calculate average confidence over the buffer
                        avg_confidence = sum(prediction_buffer[face_id]) / len(prediction_buffer[face_id])
                        
                        # Determine recognition based on average confidence
                        if avg_confidence > threshold:
                            # Draw green bounding box for recognized face
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Display name and confidence
                            conf_text = f"{student_id} ({avg_confidence:.2f})"
                            text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(display_frame, conf_text, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
                            # Mark attendance if consistently recognized with high confidence
                            if avg_confidence > CONFIDENCE_THRESHOLD and student_id not in recognized_students:
                                if mark_attendance(student_id, avg_confidence):
                                    recognized_students.add(student_id)
                                    status_text.success(f"✅ Attendance marked for {student_id}")
                        else:
                            # Draw red bounding box for unrecognized face
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Unknown ({avg_confidence:.2f})", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except Exception as e:
                        # If processing this specific face fails, continue with next face
                        continue
        
        except Exception as e:
            status_text.error(f"Error during face detection: {str(e)}")
            continue

        # Display metrics
        fps_metric.metric("FPS", f"{fps:.1f}")

if __name__ == "__main__":
    main()
