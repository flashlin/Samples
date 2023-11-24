import json
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_lit import load_markdown_documents
from web_crawler_lit import download_html, convert_html_body_to_markdown
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


def load_llm_model(model_name):
    return LlamaCpp(
        model_path=model_name,
        temperature=0.75,
        max_tokens=2000,
        top_p=3,
        n_ctx=1024 * 16,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=True,
        n_gpu_layers=52,
        n_threads=8,
    )

llm = None

def generate_questions_from_markdown(content: str):
    global llm
    prompt = """
    ```
    {content}
    ```
    Read the content above, identify entities, such as keywords, title, subject, people, events, things, date, numbers, and times.
    Generate 10 different questions based on the content about these entities, avoiding repetition of the same entity. 
    List the 10 english questions directly.
    """
    instruction = prompt.format(content=markdown)
    questions_content = llm(instruction)
    return questions_content


def get_answer_from_content(content: str, question: str):
    global llm
    prompt = """
    ```
    {content}
    ```
    Read the content above, My Question is `{question}`
    Answer the questions based on the content above. 
    If the answer is not found in the content, try to think of an answer yourself. 
    If you do not know the answer, simply answer `None`. Do not try to create false answers.
    """
    instruction = prompt.format(content=content, question=question)
    answer = llm(instruction)
    return answer


def extrac_question_body(question_line: str):
    match = re.match(f'\d+. (.*)', question_line)
    if match:
        return match.group(1).strip()
    return None


def split_questions_content(content: str):
    question_lines = re.findall(r'\d+\. .*', content)
    questions = []
    for question_line in question_lines:
        q = extrac_question_body(question_line)
        questions.append(q)
    return questions


def append_to_jsonl(question: str, answer: str):
    with open('./results/llm-qa.jsonl', 'a', encoding='utf-8') as f:
        qa_json = json.dumps({
            'instruction': question,
            'input': '',
            'output': answer
        })
        f.write(qa_json + '\r\n')


def append_to_md(question: str, answer: str):
    with open('./results/llm-qa.md', 'a', encoding='utf-8') as f:
        f.write(f'Question: {question}\r\n')
        f.write(f'Answer: {answer}\r\n')





def clean_file(file: str):
    print(f"clean {file}...")
    with open(file, "r", encoding='utf-8') as f:
        content = f.read()
    index = content.find("This article is also available in the following languages:")
    if index != -1:
        print(f"{file} clean done.")
        new_content = content[:index]
        with open(file, "w", encoding='utf-8') as f:
            f.write(new_content)

def clean_files(folder: str):
    file_names = os.listdir(folder)
    for file_name in file_names:
        clean_file(f"{folder}/{file_name}")


if __name__ == '__main__':
    clean_files('./data')

    llm = load_llm_model('../models/neural-chat-7b-v3-16k-q4_k_m.gguf')
    # html = download_html('https://ithelp.ithome.com.tw/articles/10335513')
    # markdown = convert_html_body_to_markdown(html)

    documents = load_markdown_documents('./data')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 * 10, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    for doc in all_splits:
        markdown = doc.page_content
        source = doc.metadata['source']

        # markdown = """
        # PageAttention’s memory sharing greatly reduces the memory overhead of complex sampling algorithms, such as parallel sampling and beam search, cutting their memory usage by up to 55%. This can translate into up to 2.2x improvement in throughput. This makes such sampling methods practical in LLM services.
        # PagedAttention is the core technology behind vLLM, our LLM inference and serving engine that supports a variety of models with high performance and an easy-to-use interface. For more technical details about vLLM and PagedAttention, check out our GitHub repo and stay tuned for our paper.
        # """

        print(f"generate questions for {source}")
        questions_content = generate_questions_from_markdown(markdown)
        questions = split_questions_content(questions_content)
        for question in questions:
            print(f"ask {question} for {source}")
            answer = get_answer_from_content(markdown, question)
            answer = answer.strip()
            if answer != 'None':
                append_to_jsonl(question, answer)
                append_to_md(question, answer)
