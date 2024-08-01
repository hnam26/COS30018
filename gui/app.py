import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from tool import stream_data, get_response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a chat interface
st.title("Chatbot")

# Allow the user to enter the context manually
user_context = st.text_area("Enter the context")

if "context" not in st.session_state or st.session_state.context != user_context:
    # The context has changed, so reset the chat history
    st.session_state.chat_history = []
    st.session_state.context = user_context

context = st.session_state.context

st.write("Context:", context)  # Print the context to the screen

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

if context != '':
    user_query = st.chat_input("Your message")

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            ai_response = get_response(user_query, context)  # Pass the context to the get_response function
            st.write_stream(stream_data(ai_response))
            st.session_state.chat_history.append(AIMessage(ai_response))