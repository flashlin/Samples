from langchain.vectorstores import Chroma


def load_chroma_from_documents(texts, embedding, persist_directory='db'):
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    return vectordb
