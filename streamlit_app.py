import streamlit as st
from llm_chain import create_llm_chain

# Streamlit app layout
st.title("Chatbot met LangChain en Streamlit")

# Maak een LLMChain instantie
@st.cache_resource
def get_llm_chain():
    print("Creating new LLMChain instance")  # Debug print
    return create_llm_chain()

llm_chain = get_llm_chain()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def run_chatbot():
    user_input = st.text_input("Jouw vraag:", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Genereer een antwoord met de LLMChain
        response = llm_chain.predict(question=user_input, history=st.session_state.messages)

        # Voeg het antwoord van de AI toe aan de sessiegeschiedenis
        st.session_state.messages.append({"role": "ai", "content": response})

    # Toon de chatgeschiedenis in Streamlit
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Fredbot:** {message['content']}")
    
    # Voeg een tekstblok toe om de huidige memory inhoud te tonen
    st.text_area("Current Memory Content:", value=str(llm_chain.memory.load_memory_variables({})), height=200, disabled=False)

    # Voeg een knop toe om de memory te wissen
    if st.button("Wis Memory"):
        llm_chain.memory.clear()
        st.session_state.messages = []
        st.success("Memory en chatgeschiedenis zijn gewist!")

run_chatbot()
