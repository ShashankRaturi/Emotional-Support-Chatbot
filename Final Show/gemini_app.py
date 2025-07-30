import streamlit as st
from gemini_predict import predict_emotion
from gemini_response import generate_empathetic_response
import matplotlib.pyplot as plt

# Initialize session state to store predicted emotions
if "emotions" not in st.session_state:
    st.session_state.emotions = []

st.set_page_config(page_title="Emotion Support Bot", layout="centered")
st.title("Empathetic Emotion-Aware Chatbot")
st.markdown("Enter how you're feeling and receive a thoughtful response.")

# User input box
user_input = st.text_area("Your Message", height=200)

# process on button click
if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Please wait , while we analyzing..."):
            
            # predict emotion
            emotion = predict_emotion(user_input)

            # generate response
            response = generate_empathetic_response(user_input , emotion ,max_new_tokens = 60)

        # save emotion
        st.session_state.emotions.append(emotion)

        # Display output
        st.markdown(f"**Predicted Emotion:** `{emotion}`")
        st.markdown("### 💬 Empathetic Response:")
        st.success(response)
    else:
        st.warning("Please enter a message before generating a response.")


# Handle End of Conversation
if st.button("End Conversation and Show Mood Graph"):
    if st.session_state.emotions:
        # Create a simple graph
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(st.session_state.emotions) + 1),
                st.session_state.emotions,
                marker='o', linestyle='-', color='teal')
        ax.set_title("Mood Variation During Conversation")
        ax.set_xlabel("Message Index")
        ax.set_ylabel("Predicted Emotion")
        ax.set_yticks(range(len(set(st.session_state.emotions))))
        ax.set_yticklabels(list(set(st.session_state.emotions)))
        ax.grid(True)

        st.pyplot(fig)

        # Clear emotions list after showing the graph
        st.session_state.emotions = []
    else:
        st.info("No messages yet to show a mood graph.")