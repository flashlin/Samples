import random
import time

import gradio as gr


def custom_predict(chatbot, question):
    answer = f"hi {question=} Hi" #qa(context=context, question=question)
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    chatbot.append((answer, bot_message))
    time.sleep(2)
    return chatbot, ""


def main():
    CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """

    with gr.Blocks() as demo:
        with gr.Group():
            chatbot = gr.Chatbot(label="Chatbot", height=800)
            with gr.Row():
                question = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="Type a message...",
                    scale=10,
                    lines=20
                )
                submit_button = gr.Button(
                    "Submit", variant="primary", scale=1, min_width=0
                )
                submit_button.style.align = "start"
        submit_button.click(fn=custom_predict, inputs=[chatbot, question], outputs=[chatbot, question])

    demo.queue(max_size=20).launch(server_port=5002, share=False)


if __name__ == '__main__':
    main()
