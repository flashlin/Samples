# Creating qdrant client
from typing import Dict, Union, List

from qdrant_client import QdrantClient

client = QdrantClient(url="***", api_key="***")

# Wrapping RetrieveUserProxyAgent
from litellm import embedding as test_embedding
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from qdrant_client.models import SearchRequest, Filter, FieldCondition, MatchText


class QdrantRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def query_vector_db(self, query_texts: List[str], n_results: int = 10,
                        search_string: str = "", **kwargs) -> Dict[str, Union[List[str], List[List[str]]]]:
        # define your own query function here
        embed_response = test_embedding('text-embedding-ada-002', input=query_texts)

        all_embeddings: List[List[float]] = []
        for item in embed_response['data']:
            all_embeddings.append(item['embedding'])

        search_queries: List[SearchRequest] = []
        for embedding in all_embeddings:
            search_queries.append(
                SearchRequest(
                    vector=embedding,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="page_content",
                                match=MatchText(
                                    text=search_string,
                                )
                            )
                        ]
                    ),
                    limit=n_results,
                    with_payload=True,
                )
            )

        search_response = client.search_batch(
            collection_name="{your collection name}",
            requests=search_queries,
        )

        return {
            "ids": [[scored_point.id for scored_point in batch] for batch in search_response],
            "documents": [[scored_point.payload.get('page_content', '') for scored_point in batch] for batch in search_response],
            "metadatas": [[scored_point.payload.get('metadata', {}) for scored_point in batch] for batch in search_response]
        }

    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = "", **kwargs):
        results = self.query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            **kwargs,
        )

        self._results = results


# Use QdrantRetrieveUserProxyAgent
qdrant_rag_agent = QdrantRetrieveUserProxyAgent(
    name="rag_proxy_agent",
    human_input_mode="NEVER",
    is_termination_msg=None,
    retrieve_config={
        "task": "qa",
    },
    kwargs={"max_consecutive_auto_reply": 2}
)

qdrant_rag_agent.retrieve_docs("What is Autogen?", n_results=10, search_string="autogen")
