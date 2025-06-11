import torch
from pathlib import Path

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {DEVICE}")

# If GPU is available, print details
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
ATTENDANCE_DIR = BASE_DIR / "data" / "attendance_records"

# Model paths
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Face detection settings
FACE_DETECTION_THRESHOLD = 0.9
RECOGNITION_THRESHOLD = 0.7
IMAGE_SIZE = 160  # Facenet requires 160x160 images