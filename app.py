# import streamlit as st
# import youtube_assistant as yap

# st.title("YouTube Assistant")

# question = st.chat_input("Ask a question about the video")

# if question:
#     st.write("You:", question)

#     with st.spinner("Getting an answer..."):
#         answer = yap.ask_video(question)

#     st.write("Assistant:", answer)

import streamlit as st
import youtube_assistant as yap

st.title("YouTube Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Redisplay the conversation after Streamlit reruns the app.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the video")

if question:
    # Save and display the new question.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Give the agent the earlier conversation as context.
    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Getting an answer..."):
            answer = yap.ask_video(question, history)
        st.markdown(answer)

    # Save the answer so it appears after the next rerun.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )