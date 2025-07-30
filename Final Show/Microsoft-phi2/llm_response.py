from transformers import AutoTokenizer , AutoModelForCausalLM , pipeline

def load_phi2_pipeline():
    model_id = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = "auto"
    )

    pipe = pipeline(
        "text-generation",
        model = model,        
        tokenizer = tokenizer,
        device = -1 # force CPU
    )

    return pipe


def generate_empathetic_response(user_input , emotion , pipeline , max_new_tokens):

    prompt = (
        f"You are an empathetic mental health assistant.\n"
        f"A user is feeling {emotion} and shared the following message:\n\n"
        f"\"{user_input}\"\n\n"
        f"How would you respond to provide comfort and support?\n\n"
        f"Response:"
    )

    response = pipeline(
        prompt , 
        do_sample = True , 
        temperature = 0.7,
        max_new_tokens = max_new_tokens
    
    )

    generated_text = response[0]['generated_text']
    if  "Response:" in generated_text:
        return generated_text.split("Response:")[-1].strip()
    return generated_text.strip()
