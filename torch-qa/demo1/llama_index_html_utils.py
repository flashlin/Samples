from llama_index.readers.file.flat_reader import FlatReader
from pathlib import Path
from llama_index.node_parser import UnstructuredElementNodeParser
import pickle

node_parser = UnstructuredElementNodeParser()
raw_nodes = node_parser.get_nodes_from_documents(docs)
#pickle.dump(raw_nodes, open('xx.pkl', 'wb'))
#raw_nodes = pickle.load(open('xxx.pkl', 'rb'))
base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(raw_nodes)
