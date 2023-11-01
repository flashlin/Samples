from llama_index.prompts import PromptTemplate
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from pathlib import Path
from llama_index import download_loader

PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('/work/abc.pdf'))


selected_model = LLAMA2_13B_CHAT

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.75,
    max_new_tokens=2000,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024,
                                               embed_model="local")

index = VectorStoreIndex.from_documents(documents,service_context=service_context, show_progress=True)
query_engine = index.as_query_engine()