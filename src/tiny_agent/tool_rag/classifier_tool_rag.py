import os
from typing import Any, Sequence

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.tiny_agent.models import TinyAgentToolName
from src.tiny_agent.tool_rag.base_tool_rag import BaseToolRAG, ToolRAGResult
from src.tools.base import StructuredTool, Tool


class ClassifierToolRAG(BaseToolRAG):
    _CLASSIFIER_MODEL_NAME = "squeeze-ai-lab/TinyAgent-ToolRAG"
    _DEFAULT_TOOL_THRESHOLD = 0.5
    _NUM_LABELS = 16
    _ID_TO_TOOL = {
        0: TinyAgentToolName.CREATE_CALENDAR_EVENT,
        1: TinyAgentToolName.GET_PHONE_NUMBER,
        2: TinyAgentToolName.GET_EMAIL_ADDRESS,
        3: TinyAgentToolName.OPEN_AND_GET_FILE_PATH,
        4: TinyAgentToolName.SUMMARIZE_PDF,
        5: TinyAgentToolName.COMPOSE_NEW_EMAIL,
        6: TinyAgentToolName.REPLY_TO_EMAIL,
        7: TinyAgentToolName.FORWARD_EMAIL,
        8: TinyAgentToolName.MAPS_OPEN_LOCATION,
        9: TinyAgentToolName.MAPS_SHOW_DIRECTIONS,
        10: TinyAgentToolName.CREATE_NOTE,
        11: TinyAgentToolName.OPEN_NOTE,
        12: TinyAgentToolName.APPEND_NOTE_CONTENT,
        13: TinyAgentToolName.CREATE_REMINDER,
        14: TinyAgentToolName.SEND_SMS,
        15: TinyAgentToolName.GET_ZOOM_MEETING_LINK,
    }

    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    _classifier_model: Any
    _tool_threshold: float

    def __init__(
        self,
        embedding_model: (
            AzureOpenAIEmbeddings | OpenAIEmbeddings | HuggingFaceEmbeddings
        ),
        tools: Sequence[Tool | StructuredTool],
        tool_threshold: float = _DEFAULT_TOOL_THRESHOLD,
    ):
        super().__init__(embedding_model, tools)

        self._tokenizer = AutoTokenizer.from_pretrained(
            ClassifierToolRAG._CLASSIFIER_MODEL_NAME
        )
        self._classifier_model = AutoModelForSequenceClassification.from_pretrained(
            ClassifierToolRAG._CLASSIFIER_MODEL_NAME,
            num_labels=ClassifierToolRAG._NUM_LABELS,
        )
        self._tool_threshold = tool_threshold

    @property
    def tool_rag_type(self) -> str:
        return "classifier_tool_rag"

    def retrieve_examples_and_tools(self, query: str, top_k: int) -> ToolRAGResult:
        """
        Returns the in-context examples as a formatted prompt and the tools that are relevant to the query.
        It first retrieves the best tools for the given query and then it retrieves top k examples
        that use the retrieved tools.
        """
        retrieved_tools = self._classify_tools(query)
        # Filter the tools that are available
        retrieved_tools = list(set(retrieved_tools) & set(self._available_tools))
        filtered_embeddings = self._load_filtered_embeddings(retrieved_tools)
        retrieved_embeddings = self._retrieve_top_k_embeddings(
            query, filtered_embeddings, top_k
        )

        in_context_examples_prompt = BaseToolRAG._get_in_context_examples_prompt(
            retrieved_embeddings
        )

        return ToolRAGResult(
            in_context_examples_prompt=in_context_examples_prompt,
            retrieved_tools_set=retrieved_tools,
        )

    def _classify_tools(self, query: str) -> list[TinyAgentToolName]:
        """
        Retrieves the best tools for the given query by classification.
        """
        inputs = self._tokenizer(
            query, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Get the output probabilities
        with torch.no_grad():
            outputs = self._classifier_model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)

        # Retrieve the tools that have a probability greater than the threshold
        retrieved_tools = [
            ClassifierToolRAG._ID_TO_TOOL[i]
            for i, prob in enumerate(probs[0])
            if prob > self._tool_threshold
        ]

        return retrieved_tools
