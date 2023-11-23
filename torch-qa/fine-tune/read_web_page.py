import json
import re

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

llm = load_llm_model('../models/neural-chat-7b-v3-16k-q4_k_m.gguf')

def generate_questions_from_markdown(content: str):
    global llm
    prompt = """
    ```
    {content}
    ```
    Read the content above, identify entities, such as keywords, people, events, things, numbers, and times.
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
    with open('llm-qa.jsonl', 'a', encoding='utf-8') as f:
        qa_json = json.dumps({
            'instruction': question,
            'input': '',
            'output': answer
        })
        f.write(qa_json + '\r\n')


def append_to_md(question: str, answer: str):
    with open('llm-qa.md', 'a', encoding='utf-8') as f:
        f.write(f'Question: {question}\r\n')
        f.write(f'Answer: {answer}\r\n')


if __name__ == '__main__':

    # html = download_html('https://ithelp.ithome.com.tw/articles/10335513')
    # markdown = convert_html_body_to_markdown(html)

    markdown = """
    PageAttention’s memory sharing greatly reduces the memory overhead of complex sampling algorithms, such as parallel sampling and beam search, cutting their memory usage by up to 55%. This can translate into up to 2.2x improvement in throughput. This makes such sampling methods practical in LLM services.
    PagedAttention is the core technology behind vLLM, our LLM inference and serving engine that supports a variety of models with high performance and an easy-to-use interface. For more technical details about vLLM and PagedAttention, check out our GitHub repo and stay tuned for our paper.
    """

    questions_content = generate_questions_from_markdown(markdown)

#     questions_content = """1. What is PageAttention?
# 2. How does PageAttention reduce memory overhead in complex sampling algorithms like parallel sampling and beam search?
# 3. In what aspect can significant performance improvements be seen by using PagedAttention?
# 4. What is the connection between PagedAttention and vLLM in terms of LLM models' support and inference engine?
# 5. How does GitHub repo provide further information about vLLM and its technology?
# 6. What are some common algorithms that benefit from PageAttention's memory sharing feature, particularly mentioned in the context of language models?
# 7. In comparison to traditional methods, how much memory usage reduction is achieved by PagedAttention for parallel sampling and beam search?
# 8. How does this improvement affect the practicality of LLM services during complex tasks like inference or serving engines?
# 9. What specific functions are offered within vLLM's approach to handling various models efficiently?
# 10. What aspects of PageAttention allow it to optimize language processing techniques like parallel sampling and beam search?
#     """.strip()
    questions = split_questions_content(questions_content)

    for question in questions:
        answer = get_answer_from_content(markdown, question)
        answer = answer.strip()
        if answer != 'None':
            append_to_jsonl(question, answer)
            append_to_md(question, answer)
