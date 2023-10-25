from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Payload
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from torch.utils.data import Dataset

from lanchainlit import load_txt_documents

client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "Lyrics"

def connection():
    client = QdrantClient("http://localhost:6333")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            distance=models.Distance.COSINE,
            size=1536),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
    )
    return client


def get_embedding(text, model_name):
    embedding_model_name = "BAAI/bge-base-en"
    response = openai.Embedding.create(
        input=text,
        engine=model_name
    )
    return response['data'][0]['embedding']


def upsert_vector(client, vectors, data):
    for i, vector in enumerate(vectors):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=i,
                                vector=vectors[i],
                                payload=data[i])]
        )

    print("upsert finish")


def search_from_qdrant(client, vector, k=1):
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=k,
        append_payload=True,
    )
    return search_result

def get_document_store(docs, embeddings):
    return Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name="PDF_Langchain",
        force_recreate=True
    )

def ask_main():
    embeddings = get_embeddings()
    docs = load_and_split_documents()
    doc_store = get_document_store(docs, embeddings)
    llm = get_chat_model()

    # 在 main() 加上與修改下面這段
    QUESTION_PROMPT = PromptTemplate.from_template("""你是生育補貼小幫手，只能回答生育補貼的問題，其他問題一律不回答。並以繁體中文回答問題。""")

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_store.as_retriever(),
        condense_question_prompt=QUESTION_PROMPT, #加上這段, 就不會回答本文以外的問題
        return_source_documents=True,
        verbose=False
    )

    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)


def ask_question_with_context(qa, question, chat_history):
    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history




class LlmEmbeddings:
    def __init__(self):
        # model_name = "BAAI/bge-large-en"
        # encode_kwargs = {'normalize_embeddings': False}
        # hf = HuggingFaceBgeEmbeddings(
        #     model_name=model_name,
        #     encode_kwargs=encode_kwargs
        # )
        """
        import chromadb
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
        """
        embedding_model_name = "../models/BAAI_bge-base-en"
        encode_kwargs = { 'normalize_embeddings': True }  # set True to compute cosine similarity
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs
        )

    def get_embeddings(self, text: str) -> list[float]:
        return self.embedding.embed_query(text)

    def get_web_links(self, web_links: list[str]):
        loader = WebBaseLoader(web_links)
        web_documents = loader.load()
        return web_documents

    def get(self):
        load_un_sdg_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=hf)
        load_paris_agreement_store = Chroma(persist_directory="store/paris_chroma_cosine", embedding_function=hf)
        retriever_un_sdg = load_un_sdg_store.as_retriever(search_type="similarity",
                                                          search_kwargs={"k": 3, "include_metadata": True})

        retriever_paris_agreement = load_paris_agreement_store.as_retriever(search_type="similarity",
                                                                            search_kwargs={"k": 3,
                                                                                           "include_metadata": True})
        lotr = MergerRetriever(retrievers=[retriever_un_sdg, retriever_paris_agreement])
        docs = lotr.get_relevant_documents(query)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        # Confirm that the 4 relevant documents are at beginning and end.
        reordered_docs



class QdrantRetriever:
    def __init__(self, client: QdrantClient, llm_embeddings):
        self.client = client
        self.embeddings = llm_embeddings

    def create_collection(self, collection_name: str):
        dim = 1536
        dim = 768
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE,
                size=dim),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
        )

    def upsert(self, collection_name: str, dataset: Dataset):
        payloads = dataset.select_columns(["label_names", "text"]).to_pandas().to_dict(orient="records")
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=dataset["idx"],
                vectors=dataset["embedding"],
                payloads=payloads
            )
        )

    def upsert_docs(self, collection_name: str, docs: list[Document]):
        ids = []
        vectors = []
        payloads = []
        for idx, doc in enumerate(docs):
            embeddings = self.embeddings.get_embeddings(doc.page_content)
            ids.append(idx)
            vectors.append(embeddings)
            payload = {
                'page_content': doc.page_content,
                'source': doc.metadata['source']
            }
            payloads.append(payload)
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads #[Payload(payload=point.payload) for point in docs_store]
            )
        )
        print("upsert done")

    def get_retriever(self, collection_name: str):
        collection = self.client.get_collection(collection_name)
        return collection.as_retriever()

    def search(self, collection_name, query: str, k=3):
        query_embedding = self.embeddings.get_embeddings(query)
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            append_payload=True,
        )
        return search_result


def main1():
    EMBEDDING_MODEL_NAME = "embedding-ada-002"
    openai.api_base = "https://japanopenai2023ironman.openai.azure.com/"
    openai.api_key = "yourkey"
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    qclient = connection()

    data_objs = [
        {
            "id": 1,
            "lyric": "我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離"
        },
        {
            "id": 2,
            "lyric": "而我，在這座城市遺失了你，順便遺失了自己，以為荒唐到底會有捷徑。而我，在這座城市失去了你，輸給慾望高漲的自己，不是你，過分的感情"
        }
    ]
    embedding_array = [get_embedding(text["lyric"], EMBEDDING_MODEL_NAME)
                       for text in data_objs]


    upsert_vector(qclient, embedding_array, data_objs)

    query_text = "工程師寫城市"
    query_embedding = get_embedding(query_text, EMBEDDING_MODEL_NAME)
    results = search_from_qdrant(qclient, query_embedding, k=1)
    print(f"尋找 {query_text}:", results)

    #---------
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                              deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                              openai_api_type="azure",
                              chunk_size=1)

    text_array = ["我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離",
                "而我，在這座城市遺失了你，順便遺失了自己，以為荒唐到底會有捷徑。而我，在這座城市失去了你，輸給慾望高漲的自己，不是你，過分的感情"]

    doc_store = Qdrant.from_texts(
        text_array, embeddings, url="http://localhost:6333", collection_name="Lyrics_Langchain")


    question = "工程師寫程式"
    docs = doc_store.similarity_search_with_score(question)

    document, score = docs[0]
    print(document.page_content)
    print(f"\nScore: {score}")


def main():
    llm_embeddings = LlmEmbeddings()
    resp = llm_embeddings.get_embeddings("How to use C# write HELLO")
    print(f"{len(resp)=}")
    docs = load_txt_documents("../data")
    retriever = QdrantRetriever(client, llm_embeddings)
    retriever.create_collection('sample')
    retriever.upsert_docs('sample', docs)
    result = retriever.search('sample', 'How to create pinia store in vue3?')
    print(f"{result=}")

if __name__ == '__main__':
    main()






