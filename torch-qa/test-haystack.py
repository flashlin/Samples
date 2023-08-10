from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.nodes import PromptNode, PromptModel
from haystack.agents.conversational import ConversationalAgent
from haystack.nodes import EmbeddingRetriever, PreProcessor, TextConverter, FileTypeClassifier
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http


# https://docs.haystack.deepset.ai/docs/retriever

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
with open('d:/demo/huggingface-api-key.txt', 'r', encoding='utf-8') as f:
    hf_token = f.readline()

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, use_auth_token=hf_token)
# disable Tensor Parallelism (https://github.com/huggingface/transformers/pull/24906)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)


#--------------------------------------------------------------------
# try create document
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
doc_dir = "data"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(docs)

# sentence-transformers/multi-qa-mpnet-base-dot-v1
# sentence-transformers/all-mpnet-base-v2
# document_store = InMemoryDocumentStore()
embedding_retriever = EmbeddingRetriever(document_store=document_store,
                                         embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                         model_format="sentence_transformers",
                                         top_k=10)
document_store.update_embeddings(embedding_retriever)
file_classifier = FileTypeClassifier()
text_converter = TextConverter()
# pdf_converter = PDFConverter()
# preprocessor = PreProcessor(split_by="word", split_length=250, split_overlap=30,
#                             split_sentence_boundary=True,
#                             language="en")
# indexing_pipeline = Pipeline()
# indexing_pipeline.add_node(component=file_classifier, name="FileTypeClassifier", inputs=["File"])
# indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
# # indexing_pipeline.add_node(component=pdf_converter, name="PDFConverter", inputs=["FileTypeClassifier.output_2"])
# indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter", "PDFConverter"])
# indexing_pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["PreProcessor"])
# indexing_pipeline.add_node(component=document_store, name="InMemoryDocumentStore", inputs=["Retriever"])

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, embedding_retriever)
while True:
    user_input = input('query document: ')
    if user_input == 'q':
        break
    prediction = pipe.run(
        query="How to add new b2b2c domain?",
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )
    print_answers(prediction, details="minimum")


# inspiration: https://docs.haystack.deepset.ai/docs/prompt_node#using-models-not-supported-in-hugging-face-transformers
pn = PromptNode(MODEL_NAME,
                max_length=1000,
                model_kwargs={'model': model,
                              'tokenizer': tokenizer,
                              'task_name': 'text2text-generation',
                              'device': None,  # placeholder needed to make the underlying HF Pipeline work
                              'stream': True})

# quick sanity check
# input_text = "Describe the solar system."
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# outputs = model.generate(input_ids, max_length=50)
# print(tokenizer.decode(outputs[0]))

# simply call the PromptNode
# pn("What's the coolest city in Italy? Explain reasons why")

prompt_template = """
[INST] <>
You are a helpful assistant who writes short answers.
<>\n\n
{memory} [INST] {query} [/INST]
"""



conversational_agent = ConversationalAgent(
    prompt_node=pn,
    prompt_template=prompt_template,
)


while True:
    query = input("\nHuman (type 'exit' or 'quit' to quit): ")
    if query.lower() == "exit" or query.lower() == "quit":
        break
    conversational_agent.run(query)
