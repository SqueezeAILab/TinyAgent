from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.llms import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.utils.logger_utils import log

DEFAULT_SAFE_CONTEXT_LENGTH = 512
DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE = 128


def get_model(
    model_type,
    model_name,
    api_key,
    vllm_port,
    stream,
    temperature=0,
    azure_endpoint=None,
    azure_deployment=None,
    azure_api_version=None,
):
    if model_type == "openai":
        if api_key is None:
            raise ValueError("api_key must be provided for openai model")
        llm = ChatOpenAI(
            model_name=model_name,  # type: ignore
            openai_api_key=api_key,  # type: ignore
            streaming=stream,
            temperature=temperature,
        )

    elif model_type == "vllm":
        if vllm_port is None:
            raise ValueError("vllm_port must be provided for vllm model")
        if stream:
            log(
                "WARNING: vllm does not support streaming. "
                "Setting stream=False for vllm model."
            )
        llm = OpenAI(
            openai_api_base=f"http://localhost:{vllm_port}/v1",
            model_name=model_name,
            temperature=temperature,
            max_retries=1,
            streaming=stream,
        )
    elif model_type == "local":
        if vllm_port is None:
            raise ValueError("vllm_port must be provided for vllm model")
        llm = ChatOpenAI(
            openai_api_key=api_key,  # type: ignore
            openai_api_base=f"http://localhost:{vllm_port}/v1",
            model_name=model_name,
            temperature=temperature,
            max_retries=1,
            streaming=stream,
        )
    elif model_type == "azure":
        if api_key is None:
            raise ValueError("api_key must be provided for azure model")
        if azure_api_version is None:
            raise ValueError("azure_api_version must be provided for azure model")
        if azure_endpoint is None:
            raise ValueError("azure_endpoint must be provided for azure model")
        if azure_deployment is None:
            raise ValueError("azure_deployment must be provided for azure model")

        llm = AzureChatOpenAI(
            api_key=api_key,  # type: ignore
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            streaming=stream,
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return llm


def get_embedding_model(
    model_type: str,
    model_name: str,
    api_key: str,
    azure_embedding_deployment: str,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    local_port: int | None,
    context_length: int | None,
) -> OpenAIEmbeddings | AzureOpenAIEmbeddings | HuggingFaceEmbeddings:
    if model_name is None:
        raise ValueError("Embedding model's model_name must be provided")

    if model_type == "openai":
        if api_key is None:
            raise ValueError("api_key must be provided for openai model")
        return OpenAIEmbeddings(api_key=api_key, model=model_name)
    elif model_type == "azure":
        if api_key is None:
            raise ValueError("api_key must be provided for azure model")
        if azure_api_version is None:
            raise ValueError("azure_api_version must be provided for azure model")
        if azure_endpoint is None:
            raise ValueError("azure_endpoint must be provided for azure model")
        return AzureOpenAIEmbeddings(
            api_key=api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_embedding_deployment,
            model=model_name,
        )
    elif model_type == "local":
        if local_port is None:
            # Use SentenceTransformer for local embeddings
            return HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"batch_size": DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE},
            )
        if context_length is None:
            print(
                "WARNING: context_length not provided for local model. Using default value (512).",
                flush=True,
            )
            context_length = DEFAULT_SAFE_CONTEXT_LENGTH
        return OpenAIEmbeddings(
            api_key=api_key,
            base_url=f"http://localhost:{local_port}/v1",
            model=model_name,
            embedding_ctx_length=context_length - 1,
            tiktoken_enabled=False,
            tiktoken_model_name=model_name,
        )
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
