import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, MODEL_DIR, 
    IMAGE_SIZE, DEVICE, FACE_DETECTION_THRESHOLD
)

def process_faces():
    """Process raw face images, align them, and save embeddings."""
    # Initialize models with GPU support
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, 
        margin=20, 
        keep_all=False, 
        device=DEVICE,
        thresholds=[FACE_DETECTION_THRESHOLD]*3
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    
    embeddings = []
    labels = []
    
    student_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    for student_dir in tqdm(student_dirs, desc="Processing students"):
        student_id = student_dir.name
        image_files = list(student_dir.glob('*.jpg'))
        
        processed_student_dir = PROCESSED_DATA_DIR / student_id
        processed_student_dir.mkdir(exist_ok=True)
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                
                # Detect and align face (runs on GPU if available)
                face = mtcnn(img)
                
                if face is not None:
                    # Move face tensor to GPU
                    face = face.to(DEVICE)
                    
                    # Get embedding (runs on GPU)
                    embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()
                    embeddings.append(embedding.flatten())
                    labels.append(student_id)
                    
                    # Save processed face
                    face_img = torch.nn.functional.interpolate(
                        face.unsqueeze(0), 
                        size=IMAGE_SIZE, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    face_img = face_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    face_img = (face_img * 255).astype('uint8')
                    processed_img = Image.fromarray(face_img)
                    processed_img.save(processed_student_dir / img_path.name)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save embeddings and labels
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    with open(EMBEDDINGS_DIR / 'embeddings.npy', 'wb') as f:
        np.save(f, embeddings)
    
    with open(EMBEDDINGS_DIR / 'labels.npy', 'wb') as f:
        np.save(f, labels)
    
    return embeddings, labels

def train_classifier(embeddings, labels):
    """Train a classifier on the face embeddings."""
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, encoded_labels)
    
    # Save models
    with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"Training completed on {DEVICE}. Models saved.")

if __name__ == "__main__":
    print(f"\nStarting training process on {DEVICE.type.upper()}")
    embeddings, labels = process_faces()
    train_classifier(embeddings, labels)


















# import os
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# from pathlib import Path
# from tqdm import tqdm
# import pickle
# from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report
# import cv2
# from torchvision import transforms
# from collections import defaultdict
# from config import (
#     RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, MODEL_DIR, 
#     IMAGE_SIZE, DEVICE, FACE_DETECTION_THRESHOLD
# )

# class FaceProcessor:
#     def __init__(self):
#         # Enhanced MTCNN with better parameters
#         self.mtcnn = MTCNN(
#             image_size=IMAGE_SIZE,
#             margin=40,  # Increased margin for better alignment
#             keep_all=False,
#             post_process=False,  # Better for recognition
#             select_largest=False,  # Don't always take largest face
#             min_face_size=60,  # Minimum face size to detect
#             thresholds=[FACE_DETECTION_THRESHOLD]*3,
#             device=DEVICE
#         )
        
#         # FaceNet model with eval mode
#         self.resnet = InceptionResnetV1(
#             pretrained='vggface2',
#             classify=False,
#             device=DEVICE
#         ).eval()
        
#         # Image augmentations
#         self.augment = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#             transforms.RandomRotation(15),
#         ])
        
#         # Quality control thresholds
#         self.MIN_FACE_SIZE = 80
#         self.MAX_BLUR_VARIANCE = 150
        
#     def _check_quality(self, face_img):
#         """Check image quality before processing."""
#         # Convert to grayscale for blur detection
#         gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
#         blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
#         face_size = max(face_img.shape[:2])
        
#         return (face_size >= self.MIN_FACE_SIZE and 
#                 blur_value <= self.MAX_BLUR_VARIANCE)

#     def process_faces(self):
#         """Enhanced face processing with quality checks and augmentation."""
#         embeddings = []
#         labels = []
#         quality_stats = defaultdict(int)
        
#         student_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
        
#         for student_dir in tqdm(student_dirs, desc="Processing students"):
#             student_id = student_dir.name
#             image_files = list(student_dir.glob('*.jpg'))
#             processed_student_dir = PROCESSED_DATA_DIR / student_id
#             processed_student_dir.mkdir(exist_ok=True)
            
#             for img_path in image_files:
#                 try:
#                     img = Image.open(img_path)
                    
#                     # Apply augmentation
#                     img = self.augment(img)
                    
#                     # Detect and align face
#                     face = self.mtcnn(img)
                    
#                     if face is not None:
#                         face_img = (face.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        
#                         if self._check_quality(face_img):
#                             # Move to device and get embedding
#                             face = face.to(DEVICE)
#                             embedding = self.resnet(face.unsqueeze(0)).detach().cpu().numpy()
                            
#                             embeddings.append(embedding.flatten())
#                             labels.append(student_id)
                            
#                             # Save processed face
#                             processed_img = Image.fromarray(face_img)
#                             processed_img.save(processed_student_dir / img_path.name)
#                             quality_stats['accepted'] += 1
#                         else:
#                             quality_stats['rejected_quality'] += 1
#                     else:
#                         quality_stats['rejected_no_face'] += 1
                        
#                 except Exception as e:
#                     print(f"Error processing {img_path}: {e}")
#                     quality_stats['errors'] += 1
        
#         # Print quality statistics
#         print("\nImage Quality Statistics:")
#         for stat, count in quality_stats.items():
#             print(f"{stat.replace('_', ' ').title()}: {count}")
        
#         # Convert to numpy arrays
#         embeddings = np.array(embeddings)
#         labels = np.array(labels)
        
#         # Save embeddings and labels
#         np.save(EMBEDDINGS_DIR / 'embeddings.npy', embeddings)
#         np.save(EMBEDDINGS_DIR / 'labels.npy', labels)
        
#         return embeddings, labels

# class FaceClassifier:
#     def __init__(self):
#         self.normalizer = Normalizer(norm='l2')
#         self.le = LabelEncoder()
        
#     def train(self, embeddings, labels):
#         """Enhanced classifier training with cross-validation."""
#         # Normalize embeddings
#         embeddings = self.normalizer.fit_transform(embeddings)
        
#         # Encode labels
#         encoded_labels = self.le.fit_transform(labels)
        
#         # Split data for validation
#         X_train, X_val, y_train, y_val = train_test_split(
#             embeddings, encoded_labels, test_size=0.2, stratify=encoded_labels
#         )
        
#         # Test multiple classifiers
#         classifiers = [
#             ('SVM-RBF', SVC(kernel='rbf', probability=True, class_weight='balanced')),
#             ('SVM-Linear', SVC(kernel='linear', probability=True)),
#             ('RandomForest', RandomForestClassifier(n_estimators=200))
#         ]
        
#         best_score = 0
#         best_clf = None
#         best_name = ""
        
#         for name, clf in classifiers:
#             # Cross-validation
#             scores = cross_val_score(clf, X_train, y_train, cv=5)
#             mean_score = np.mean(scores)
            
#             # Full training
#             clf.fit(X_train, y_train)
#             val_score = clf.score(X_val, y_val)
            
#             print(f"\n{name} Performance:")
#             print(f"  Cross-val Accuracy: {mean_score:.2%}")
#             print(f"  Validation Accuracy: {val_score:.2%}")
            
#             if val_score > best_score:
#                 best_score = val_score
#                 best_clf = clf
#                 best_name = name
        
#         print(f"\nSelected {best_name} with validation accuracy: {best_score:.2%}")
        
#         # Save models
#         with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
#             pickle.dump(best_clf, f)
        
#         with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
#             pickle.dump(self.le, f)
        
#         with open(MODEL_DIR / 'normalizer.pkl', 'wb') as f:
#             pickle.dump(self.normalizer, f)
        
#         # Final evaluation report
#         y_pred = best_clf.predict(X_val)
#         print("\nClassification Report:")
#         print(classification_report(y_val, y_pred, target_names=self.le.classes_))
        
#         return best_clf

# if __name__ == "__main__":
#     print(f"\nStarting training process on {DEVICE.type.upper()}")
    
#     # Process faces with enhanced quality checks
#     processor = FaceProcessor()
#     embeddings, labels = processor.process_faces()
    
#     # Train classifier with cross-validation
#     classifier = FaceClassifier()
#     classifier.train(embeddings, labels)
