import abc
import os
import pickle
from dataclasses import dataclass
from typing import Collection, Sequence

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from typing_extensions import TypedDict

from src.tiny_agent.config import DEFAULT_OPENAI_EMBEDDING_MODEL
from src.tiny_agent.models import TinyAgentToolName
from src.tools.base import StructuredTool, Tool

TOOLRAG_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ToolRAGResult:
    in_context_examples_prompt: str
    retrieved_tools_set: Collection[TinyAgentToolName]


class PickledEmbedding(TypedDict):
    example: str
    embedding: torch.Tensor
    tools: Sequence[str]


class BaseToolRAG(abc.ABC):
    """
    The base class for the ToolRAGs that are used to retrieve the in-context examples and tools based on the user query.
    """

    _EMBEDDINGS_DIR_PATH = os.path.join(TOOLRAG_DIR_PATH)
    _EMBEDDINGS_FILE_NAME = "embeddings.pkl"

    # Embedding model that computes the embeddings for the examples/tools and the user query
    _embedding_model: AzureOpenAIEmbeddings | OpenAIEmbeddings | HuggingFaceEmbeddings
    # The set of available tools so that we do an initial filtering based on the tools that are available
    _available_tools: Sequence[TinyAgentToolName]
    # The path to the embeddings.pkl file
    _embeddings_pickle_path: str

    def __init__(
        self,
        embedding_model: (
            AzureOpenAIEmbeddings | OpenAIEmbeddings | HuggingFaceEmbeddings
        ),
        tools: Sequence[Tool | StructuredTool],
    ) -> None:
        self._embedding_model = embedding_model
        self._available_tools = [TinyAgentToolName(tool.name) for tool in tools]

        # TinyAgent currently only supports "text-embedding-3-small" model by default.
        # Hence, we only use the directory created for the default model.
        model_name = DEFAULT_OPENAI_EMBEDDING_MODEL
        self._embeddings_pickle_path = os.path.join(
            BaseToolRAG._EMBEDDINGS_DIR_PATH,
            model_name.split("/")[-1],  # Only use the last model name
            BaseToolRAG._EMBEDDINGS_FILE_NAME,
        )

    @property
    @abc.abstractmethod
    def tool_rag_type(self) -> str:
        pass

    @abc.abstractmethod
    def retrieve_examples_and_tools(self, query: str, top_k: int) -> ToolRAGResult:
        """
        Returns the in-context examples as a formatted prompt and the tools that are relevant to the query.
        """
        pass

    def _retrieve_top_k_embeddings(
        self, query: str, examples: list[PickledEmbedding], top_k: int
    ) -> list[PickledEmbedding]:
        """
        Computes the cosine similarity of each example and retrieves the closest top_k examples.
        If there are already less than top_k examples, returns the examples directly.
        """
        if len(examples) <= top_k:
            return examples

        query_embedding = torch.tensor(self._embedding_model.embed_query(query))
        embeddings = torch.stack(
            [x["embedding"] for x in examples]
        )  # Stacking for batch processing

        # Cosine similarity between query_embedding and all chunks
        cosine_similarities = torch.nn.functional.cosine_similarity(
            embeddings, query_embedding.unsqueeze(0), dim=1
        )

        # Retrieve the top k indices from cosine_similarities
        _, top_k_indices = torch.topk(cosine_similarities, top_k)

        # Select the chunks corresponding to the top k indices
        selected_examples = [examples[i] for i in top_k_indices]

        return selected_examples

    def _load_filtered_embeddings(
        self, filter_tools: list[TinyAgentToolName] | None = None
    ) -> list[PickledEmbedding]:
        """
        Loads the embeddings.pkl file that contains a list of PickledEmbedding objects
        and returns the filtered results based on the available tools.
        """
        with open(self._embeddings_pickle_path, "rb") as file:
            embeddings: dict[str, PickledEmbedding] = pickle.load(file)

        filtered_embeddings = []
        tool_names = [tool.value for tool in filter_tools or self._available_tools]
        for embedding in embeddings.values():
            # Check if all tools are available in this example
            if all(tool in tool_names for tool in embedding["tools"]):
                filtered_embeddings.append(embedding)

        return filtered_embeddings

    @staticmethod
    def _get_in_context_examples_prompt(embeddings: list[PickledEmbedding]) -> str:
        examples = [example["example"] for example in embeddings]
        examples_prompt = "###\n".join(examples)
        return f"{examples_prompt}###\n"
