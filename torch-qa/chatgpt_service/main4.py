import streamlit as st
import hashlib
from dotenv import load_dotenv, find_dotenv
from langchain.llms import LlamaCpp
from models import init_messages, llama2_prompt, convert_langchain_schema_to_dict, create_llama2
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)


# Convert Pass into hash format
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# Check password matches during login
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False


def login(username, password):
    hash = make_hashes("123")
    if check_hashes(password, hash):
        st.session_state["authentication_status"] = "Yes"
        return True
    return False


def get_answer(llm, messages) -> tuple[str, float]:
    # if isinstance(llm, ChatOpenAI):
    #     with get_openai_callback() as cb:
    #         answer = llm(messages)
    #     return answer.content, cb.total_cost
    if isinstance(llm, LlamaCpp):
        return llm(llama2_prompt(convert_langchain_schema_to_dict(messages))), 0.0


def show_login_form():
    login_form = st.empty()
    with login_form.form(key="login"):
        st.subheader('Log in to the App')
        username = st.text_input("User Name", placeholder='username')
        password = st.text_input("Password", type='password')
        submit_form = st.form_submit_button("Login")
        if submit_form:
            if login(username, password):
                login_form.empty()
            else:
                st.error("login fail")


def main():
    _ = load_dotenv(find_dotenv())
    llm = create_llama2()
    print("llm loaded")
    init_messages()

    if st.session_state["authentication_status"] is None:
        print("show login form")
        show_login_form()
        return

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

        # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # if st.session_state["authentication_status"]:
    #     try:
    #         if auth.reset_password(st.session_state["username"], 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)


if __name__ == '__main__':
    main()

