from llama_index_lit import load_llm_model, load_txt_documents, ConversationalAgent, create_vector_store_index

embedding_model_name = "./models/BAAI_bge-base-en"
llm_model_name = './models/TheBloke_Mistral-7B-Instruct-v0.1-GGUF'


def main():
    llm = load_llm_model(llm_model_name)
    docs = load_txt_documents('./docs')
    vector_store_index = create_vector_store_index(llm, docs)
    qa_bot = ConversationalAgent(llm, vector_store_index)

    while True:
        query = input("query: ")
        if query == 'quit' or query == 'q':
            exit(0)
        answer = qa_bot.ask(query)
        print(answer)
        print("")


if __name__ == "__main__":
    main()