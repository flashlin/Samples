import streamlit as st
import hashlib


# Convert Pass into hash format
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# Check password matches during login
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False


def login(placeholder, username, password):
    hash = make_hashes("123")
    print(f"login {username=} {password=} {hash=}")
    if check_hashes(password, hash):
        print("OK")
        st.session_state["authentication_status"] = "Yes"
        placeholder.empty()
        return
    print("FAIL")
    return


def main():
    placeholder = st.empty()
    with placeholder.form(key="login"):
        st.subheader('Log in to the App')
        username = st.text_input("User Name", placeholder='username')
        password = st.text_input("Password", type='password')
        submit_form = st.form_submit_button("Login")
        if submit_form:
            login(placeholder, username, password)

    # if st.session_state["authentication_status"]:
    #     try:
    #         if auth.reset_password(st.session_state["username"], 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)

main()
