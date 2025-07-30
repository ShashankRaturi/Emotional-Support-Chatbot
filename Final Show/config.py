import torch 

from pathlib import Path
## define base directory

# Path(__file__) gets the path to the current file (config.py)
# .parent moves one level up pointing to the parent directory
BASE_DIR = Path(__file__).resolve().parent 

## defining path of files dir
FILES_DIR = BASE_DIR/"files"

# define the paths
VOCAB_PATH = FILES_DIR/"vocab.json"
ENCODER_PATH = FILES_DIR/"label_encoder.pkl"
MODEL_PATH = FILES_DIR/"emotion_rnn_checkpoint.pth"


# other configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 500
MIN_LEN = 31
