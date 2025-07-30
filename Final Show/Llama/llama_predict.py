import torch
from model import RNN
from utils import encode_text
from config import *
from llama_response import generate_empathetic_response
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# load resources
with open(VOCAB_PATH) as file:
    vocab = json.load(file)

with open(ENCODER_PATH , 'rb') as file:
    encoder = pickle.load(file)


# initializing the model
model = RNN(
   vocab_size = len(vocab),
   embedding_dim = 100 , # same as training
   hidden_dim = 90 , # same as training
   output_dim = len(encoder.classes_),
   pad_index = vocab['<pad>']
)

# loading the model
checkpoint = torch.load(MODEL_PATH, map_location = DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()


# predict emotion
def predict_emotion(text):
   input_tensor = encode_text(text , vocab , MAX_LEN , PAD_INDEX = vocab['<pad>']).to(DEVICE)

   with torch.no_grad():
      
        output = model(input_tensor)
        predicted_class = torch.argmax(output , dim=1).item()
        emotion = encoder.inverse_transform([predicted_class])[0]

        return emotion



if __name__ == "__main__":
    
    predicted_emotions = []
    user_inputs = []

    while True:
        user_input = input("Enter your sentence : ")
        if user_input == "exit()":
            break
        
        # predict emotion
        emotion = predict_emotion(user_input)

        # generate response
        response = generate_empathetic_response(user_input , emotion , max_new_tokens=60)

        print("\nUser Input:", user_input)
        print("Predicted Emotion:", emotion)
        print("Empathetic Response:\n", response)

        predicted_emotions.append(emotion)
        user_inputs.append(user_input)


    if predicted_emotions:
        plt.figure(figsize=(12, 4))
        sns.set_theme(style="whitegrid")
        sns.lineplot(x=range(len(predicted_emotions)), y=predicted_emotions, marker="o")
        plt.xticks(range(len(user_inputs)), [f"input{i+1}" for i in range(len(user_inputs))], rotation=45)
        plt.title("Emotion Variation Over Conversation")
        plt.xlabel("User Turns")
        plt.ylabel("Predicted Emotion")
        plt.tight_layout()
        plt.savefig("mood_variation.png")
        plt.show()
