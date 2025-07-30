import torch 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = "/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/Final Show/files/vocab.json"
ENCODER_PATH = "/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/Final Show/files/label_encoder.pkl"
MODEL_PATH = "/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/Final Show/files/emotion_rnn_checkpoint.pth"
MAX_LEN = 500
MIN_LEN = 31
