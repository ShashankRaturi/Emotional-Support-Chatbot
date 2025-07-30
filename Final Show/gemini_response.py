import google.generativeai as genai
from config import GEMINI_API_KEY


# using gemini via api , instead of locally installing it

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
model = genai.GenerativeModel('gemini-2.5-pro')

def generate_empathetic_response(user_input , emotion , max_new_tokens = 100):
    
    prompt = (
        f"You are an empathetic mental health assistant.\n"
        f"A user is feeling {emotion} and shared the following message:\n\n"
        f"\"{user_input}\"\n\n"
        f"Briefly provide comfort and support in maximum 5 lines.\n\n"
        f"Response:"
    )

    response = model.generate_content(prompt)
    return response.text.strip()