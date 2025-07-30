import json
import pickle
import re
import torch


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
    

def encode_text(text , vocab , MAX_LEN , PAD_INDEX):
   
    tokens = clean_text(text).split()
    indices = [vocab.get(token , vocab['<unk>']) for token in tokens]

    if len(indices) < MAX_LEN:
       indices += [PAD_INDEX]*(MAX_LEN - len(indices)) # padding , inorder to make the tensor of same size
    else:
       indices = indices[:MAX_LEN]
    
    # unsqueeze(0) -> to make the shape of tensor from [MAX_LEN] to [1 , MAX_LEN]
    #  which is compatible with neural net requirement
    return torch.tensor(indices).unsqueeze(0)


