from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

embeddings = HypotheticalDocumentEmbedder.from_llm(llm,
                                                   bge_embeddings,
                                                   prompt_key="web_search"
                                                   )


prompt_template = """Please answer the user's question as a single food item
Question: {question}
Answer:"""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)
embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=bge_embeddings
)
result = embeddings.embed_query(
    "What is is McDonalds best selling item?"
)