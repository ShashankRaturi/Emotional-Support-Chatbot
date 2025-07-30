import requests

def generate_empathetic_response(user_input , emotion , max_new_tokens = 100):
    
    prompt = (
        f"You are an empathetic mental health assistant.\n"
        f"A user is feeling {emotion} and shared the following message:\n\n"
        f"\"{user_input}\"\n\n"
        f"Briefly provide comfort and support in maximum 5 lines.\n\n"
        f"Response:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model" : 'llama3',
            "prompt" : prompt,
            "temperature" : 0.7,
            "stream" : False,
            "max_tokens" : max_new_tokens
        }
    
    )

    data = response.json()
    full_output = data["response"]

    if "Response:" in full_output:
        return full_output.split("Response:")[-1].strip()

    return full_output.strip()