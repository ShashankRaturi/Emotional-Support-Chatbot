import torch
import torch.nn as nn
import json
import pickle
import re


# config for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define the RNN 
class RNN(nn.Module):
  
  def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , pad_index):
      super().__init__()

      self.embedding = nn.Embedding(vocab_size , embedding_dim , padding_idx=pad_index)
      self.dropout1 = nn.Dropout(0.4)
      self.rnn = nn.LSTM(embedding_dim , hidden_dim , batch_first=True , bidirectional=True)
      self.dropout2 = nn.Dropout(0.4)

      # The hidden dimension of the linear layer should be 2 * hidden_dim because of the bidirectional LSTM
      self.fc = nn.Linear(hidden_dim * 2 , output_dim)

  def forward(self , x):
    embedded = self.embedding(x)
    embedded = self.dropout1(embedded) # Apply dropout after embedding

    _ , (hidden , _) = self.rnn(embedded)
    
    # hidden has shape (2, batch_size, hidden_dim) for bidirectional LSTM
    # Concatenate the hidden states from the forward and backward directions
    hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
    hidden = self.dropout2(hidden) # Apply dropout before the linear layer
    return self.fc(hidden)
  

# load vocab
with open("/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/With RNN Over Google Colab/vocab.json") as file:
   vocab = json.load(file)


# load encoder
with open("/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/With RNN Over Google Colab/label_encoder.pkl" , "rb") as file:
   encoder = pickle.load(file)


# load metadata
with open("/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/With RNN Over Google Colab/meta.json") as file :
   meta = json.load(file)


MAX_LEN = meta['max_len']
PAD_INDEX = vocab['<pad>']



# initializing the model
model = RNN(
   vocab_size = len(vocab),
   embedding_dim = 100 , # same as training
   hidden_dim = 90 , # same as training
   output_dim = len(encoder.classes_),
   pad_index = PAD_INDEX
)

# loading the model
checkpoint = torch.load("/media/shashanks/3E0CD8C50CD878FB/CDAC/My work/CDAC PROJECTS/MACHINE LEARNING/Emotional-Support-Chatbot/With RNN Over Google Colab/emotion_rnn_checkpoint.pth" , map_location = DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()



# pre-processing

def clean_text(text):
   
    # convert into lower text
    text = text.lower()
    # remove any url from the text if present
    text = re.sub('\b(?:https?|ftp|ssh)://\S+', '' , text)
    # remove any non alphabet or non white space character
    text = re.sub(r"[^a-zA-Z\s]" , "" , text)
    # remove extra white spaces if any present
    text = re.sub(r"\s+" , " " , text).strip() 

    return text



def encode_text(text):
   
    tokens = clean_text(text).split()
    indices = [vocab.get(token , vocab['<unk>']) for token in tokens]

    if len(indices) < MAX_LEN:
       indices += [PAD_INDEX]*(MAX_LEN - len(indices)) # padding , inorder to make the tensor of same size
    else:
       indices = indices[:MAX_LEN]
    
    # unsqueeze(0) -> to make the shape of tensor from [MAX_LEN] to [1 , MAX_LEN]
    #  which is compatible with neural net requirement
    return torch.tensor(indices).unsqueeze(0).to(DEVICE) # shape : 


def predict_emotion(text):
   input_tensor = encode_text(text)

   with torch.no_grad():
      
        output = model(input_tensor)
        predicted_class = torch.argmax(output , dim=1).item()
        emotion = encoder.inverse_transform([predicted_class])[0]

        return emotion


if __name__ == "__main__":
    while True:
        user_input = input("Enter your sentence : ")
        if user_input == "exit()":
            break
        
        emotion = predict_emotion(user_input)
        print(f"Predicted Emotion : {emotion}\n")

   