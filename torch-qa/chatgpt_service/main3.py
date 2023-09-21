import streamlit as st
from streamlit_chat import message as post_message


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("The messages from Bot\nWith new line")


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault(
    'messages',
    []
)




st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
st.title('🦙💬 Llama 2 Chatbot')
st.markdown("Getting your Replicate API token is a simple 3-step process:      \n "
            "1.Go to [Replicate](https://replicate.com/signin/). \n "
            "2.Sign in with your GitHub account. \n "
            "3.Proceed to the API tokens page and copy your API token.")
# if 'REPLICATE_API_TOKEN' in st.secrets:
#     st.success('API key already provided!', icon='✅')
#     replicate_api = st.secrets['REPLICATE_API_TOKEN']
# else:
#     replicate_api = st.text_input('Enter Replicate API token:', type='password')
#     if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
#         st.warning('Please enter your credentials!', icon='⚠️')
#     else:
#         st.success('Proceed to entering your prompt message!', icon='👉')


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['messages'])):
        message = st.session_state['messages'][i]
        print(f"{message=}")
        role = message['role']
        is_user = role == 'user'
        post_message(message['content'], is_user=is_user, key=f"{i}_user")
        # message(
        #     st.session_state['generated'][i]['data'],
        #     key=f"{i}",
        #     allow_html=True,
        #     is_table=True if st.session_state['generated'][i]['type']=='table' else False
        # )
#st.button("Clear message", on_click=on_btn_click)


if prompt := st.chat_input(): #(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})


# streamlit run st-demo.py