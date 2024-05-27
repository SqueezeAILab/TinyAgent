import abc

from langchain.llms.base import BaseLLM
from langchain_core.messages import BaseMessage, get_buffer_string

from src.tiny_agent.config import ModelConfig
from src.tiny_agent.models import Tokenizer


class SubAgent(abc.ABC):
    # A constant to make the context length check more conservative
    # And to allow for more output tokens to be generated
    _CONTEXT_LENGTH_TRUST = 500

    _llm: BaseLLM
    _tokenizer: Tokenizer
    _context_length: int
    _custom_instructions: str | None

    def __init__(
        self,
        llm: BaseLLM,
        config: ModelConfig,
        custom_instructions: str | None,
    ) -> None:
        assert (
            config.tokenizer is not None
        ), "Tokenizer must be provided for sub-agents."
        self._llm = llm
        self._tokenizer = config.tokenizer
        self._context_length = config.context_length
        self._custom_instructions = custom_instructions

    @abc.abstractmethod
    async def __call__(self, *args, **kwargs) -> str:
        pass

    def check_context_length(
        self, messages: list[BaseMessage], context: str
    ) -> str | None:
        """
        Checks if the final length of the messages is greater than the context length,
        and if so, removes the excess tokens from the context. If no, return None.
        """
        text = get_buffer_string(messages)
        total_length = len(self._tokenizer.encode(text))
        length_to_remove = total_length - self._context_length

        if length_to_remove <= 0:
            return None

        context = self._tokenizer.decode(
            self._tokenizer.encode(context)[
                : -length_to_remove - self._CONTEXT_LENGTH_TRUST
            ]
        )

        return context
