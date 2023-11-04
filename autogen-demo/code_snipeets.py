import os
import autogen
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, \
    InMemoryStateManagerWithFaiss
from autollm import AutoQueryEngine
from autollm.utils.document_reading import read_github_repo_as_documents, read_files_as_documents


def load_git_repo_docs(url: str):
    # url = "https://github.com/ultralytics/ultralytics.git"
    # specify the extensions of the documents to be read
    documents = read_github_repo_as_documents(git_repo_url=url,
                                              relative_folder_path="docs",
                                              required_exts=[".md"])
    return documents

# query_engine = AutoQueryEngine.from_parameters(documents=documents)


class QaBot:
    system_prompt = "You are an friendly ai assistant that help users find the most relevant and accurate answers to their questions based on the documents you have access to. When answering the questions, mostly rely on the info in documents."
    query_wrapper_prompt = '''
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and mostly relying on it,
answer the query.
Query: {query_str}
Answer:
'''

    def __init__(self):
        self.query_engine = self.create_engine([])

    def create_lance_db_vector_store_params(self):
        vector_store_type = "LanceDBVectorStore"
        uri = "tmp/lancedb"
        table_name = "vectors"
        vector_store_params = {"vector_store_type": vector_store_type, "uri": uri, "table_name": table_name}
        return vector_store_params

    def create_engine(self, docs):
        enable_cost_calculator = True
        vector_store_params = self.create_lance_db_vector_store_params()
        model = "gpt-3.5-turbo"
        llm_params = {"model": model}
        # service context params
        chunk_size = 1024
        # query engine params
        similarity_top_k = 5
        service_context_params = {"chunk_size": chunk_size}
        query_engine_params = {"similarity_top_k": similarity_top_k}
        return AutoQueryEngine.from_parameters(documents=docs,
                                               system_prompt=self.system_prompt,
                                               query_wrapper_prompt=self.query_wrapper_prompt,
                                               enable_cost_calculator=enable_cost_calculator,
                                               llm_params=llm_params,
                                               vector_store_params=vector_store_params,
                                               service_context_params=service_context_params,
                                               query_engine_params=query_engine_params)

    def ask(self, question: str) -> str:
        response = self.query_engine.query(question)
        return response.response


config_list = [
    {
        'model': 'gpt-4'
    },
]

llm_config = {"config_list": config_list, "seed": 42}

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
)

interface = autogen_interface.AutoGenInterface()  # how MemGPT talks to AutoGen
persistence_manager = InMemoryStateManager()
persona = "I\'m a 10x engineer at a FAANG tech company."
human = "I\'m a team manager at a FAANG tech company."
memgpt_agent = presets.use_preset(presets.DEFAULT, 'gpt-4', persona, human, interface, persistence_manager)
# memgpt_agent = presets.use_preset(presets.DEFAULT_PRESET, None, 'gpt-4', persona, human, interface, persistence_manager)

# MemGPT coder
coder = memgpt_autogen.MemGPTAgent(
    name="MemGPT_coder",
    agent=memgpt_agent,
)

# non-MemGPT PM
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="First send the message 'Let's go Mario!'")
