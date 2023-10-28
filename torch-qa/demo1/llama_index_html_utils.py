from llama_index.readers.file.flat_reader import FlatReader
from pathlib import Path
from llama_index.node_parser import UnstructuredElementNodeParser
import pickle
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex


reader = FlatReader()
docs_2021 = reader.load_data(Path("tesla_2021_10k.htm"))

node_parser = UnstructuredElementNodeParser()
raw_nodes = node_parser.get_nodes_from_documents(docs_2021)
#pickle.dump(raw_nodes, open('xx.pkl', 'wb'))
#raw_nodes = pickle.load(open('xxx.pkl', 'rb'))
base_nodes_2021, node_mappings_2021 = node_parser.get_base_nodes_and_mappings(raw_nodes)

# construct top-level vector index + query engine
vector_index = VectorStoreIndex(base_nodes_2021)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# query engine without RecursiveRetriever
vector_query_engine = vector_index.as_query_engine(similarity_top_k=1)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=node_mappings_2021,
    verbose=True,
)
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

response = query_engine.query("What was the revenue in 2020?")
print(str(response))
# response.response
# response.metadata

