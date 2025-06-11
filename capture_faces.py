# import cv2
# import os
# from config import RAW_DATA_DIR
# from facenet_pytorch import MTCNN
# from PIL import Image
# import numpy as np

# def capture_faces(student_id, num_samples=20):
#     """
#     Capture face samples for a student and save to their directory.
    
#     Args:
#         student_id (str): Unique identifier for the student
#         num_samples (int): Number of face samples to capture
#     """
#     # Create student directory
#     student_dir = RAW_DATA_DIR / student_id
#     student_dir.mkdir(exist_ok=True)
    
#     # Initialize face detector
#     mtcnn = MTCNN(keep_all=True, device='cpu')
    
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise IOError("Cannot open webcam")
    
#     print(f"Capturing {num_samples} samples for student {student_id}. Press 'q' to quit.")
    
#     count = 0
#     while count < num_samples:
#         ret, frame = cap.read()
#         if not ret:
#             continue
            
#         # Convert to PIL image
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(frame_rgb)
        
#         # Detect faces
#         boxes, _ = mtcnn.detect(pil_img)
        
#         # Draw rectangles and count
#         if boxes is not None:
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Save sample when space is pressed
#             cv2.imshow('Capture Faces - Press SPACE to capture, Q to quit', frame)
#             key = cv2.waitKey(1)
            
#             if key == ord(' '):  # Space to capture
#                 # Save original frame
#                 img_path = student_dir / f"{student_id}_{count:04d}.jpg"
#                 cv2.imwrite(str(img_path), frame)
#                 print(f"Saved sample {count+1}/{num_samples}")
#                 count += 1
#             elif key == ord('q'):  # Q to quit
#                 break
#         else:
#             cv2.imshow('Capture Faces - Press SPACE to capture, Q to quit', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Completed capturing {count} samples for student {student_id}")

# if __name__ == "__main__":
#     student_id = input("Enter student ID: ")
#     capture_faces(student_id)



















import cv2
import os
from config import RAW_DATA_DIR
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from pathlib import Path

def capture_faces(student_id, num_samples=20):
    """
    Capture face samples for a student and save to their directory.

    Args:
        student_id (str): Unique identifier for the student
        num_samples (int): Number of face samples to capture
    """
    # Create student directory
    student_dir = RAW_DATA_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face detector with higher thresholds for accuracy
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.8, 0.9, 0.9],
        keep_all=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print(f"Capturing {num_samples} samples for student {student_id}. Press 'SPACE' to capture, 'Q' to quit.")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Detect faces
        boxes, probs = mtcnn.detect(pil_img)

        # Draw rectangles and count
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.98:  # filter low confidence detections
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save sample when space is pressed
            cv2.imshow('Capture Faces - Press SPACE to capture, Q to quit', frame)
            key = cv2.waitKey(1)

            if key == ord(' '):  # Space to capture
                # Save cropped face for better accuracy
                face_img = frame[y1:y2, x1:x2]
                img_path = student_dir / f"{student_id}_{count:04d}.jpg"
                cv2.imwrite(str(img_path), face_img)
                print(f"✅ Saved sample {count+1}/{num_samples}")
                count += 1
            elif key == ord('q'):
                break
        else:
            cv2.imshow('Capture Faces - Press SPACE to capture, Q to quit', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 Completed capturing {count} samples for student {student_id}")

if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    capture_faces(student_id)