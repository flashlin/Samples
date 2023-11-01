from services.langchain_lit import ConversationalRetrievalChainAgent, FaissRetrieval, LlmEmbedding, load_llm_model, load_txt_documents


embedding_model_name = "./models/BAAI_bge-base-en"
llm_model_name = './models/Mistral-7B-Instruct-v0.1-GGUF'


def main():
    llm = load_llm_model(llm_model_name)
    llm_embedding = LlmEmbedding(embedding_model_name)
    docs = load_txt_documents('./docs')
    retrieval = FaissRetrieval(llm_embedding)
    retriever = retrieval.get_retriever(docs)
    qa_bot = ConversationalRetrievalChainAgent(llm, retriever)

    while True:
        query = input("query: ")
        if query == 'quit' or query == 'q':
            exit(0)
        answer = qa_bot.ask(query)
        print(answer)
        print("")

if __name__ == "__main__":
    main()