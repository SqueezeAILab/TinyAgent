from typing import Sequence

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.tiny_agent.models import TinyAgentToolName
from src.tiny_agent.tool_rag.base_tool_rag import BaseToolRAG, ToolRAGResult
from src.tools.base import StructuredTool, Tool


class SimpleToolRAG(BaseToolRAG):
    def __init__(
        self,
        embedding_model: (
            AzureOpenAIEmbeddings | OpenAIEmbeddings | HuggingFaceEmbeddings
        ),
        tools: Sequence[Tool | StructuredTool],
    ):
        super().__init__(embedding_model, tools)

    @property
    def tool_rag_type(self) -> str:
        return "simple_tool_rag"

    def retrieve_examples_and_tools(self, query: str, top_k: int) -> ToolRAGResult:
        """
        Returns the in-context examples as a formatted prompt and the tools that are relevant to the query
        It first filters the examples based on the tools that are available and then retrieves the examples
        and tools based on the query.
        """
        filtered_embeddings = self._load_filtered_embeddings()
        retrieved_embeddings = self._retrieve_top_k_embeddings(
            query, filtered_embeddings, top_k
        )
        in_context_examples_prompt = BaseToolRAG._get_in_context_examples_prompt(
            retrieved_embeddings
        )

        tools_names = set(
            sum(
                [
                    [TinyAgentToolName(tool) for tool in example["tools"]]
                    for example in retrieved_embeddings
                ],
                [],
            )
        )

        return ToolRAGResult(
            in_context_examples_prompt=in_context_examples_prompt,
            retrieved_tools_set=tools_names,
        )
