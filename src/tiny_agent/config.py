import json
import os
from typing import Any

from tiktoken import encoding_name_for_model, get_encoding
from transformers import AutoTokenizer

from src.tiny_agent.models import (
    AgentType,
    App,
    ModelConfig,
    ModelType,
    TinyAgentConfig,
    WhisperConfig,
)

DEFAULT_SAFE_CONTEXT_LENGTH = 4096
DEFAULT_EMBEDDING_CONTEXT_LENGTH = 8192
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


OPENAI_MODELS = {
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-instruct": 4096,
}

AGENT_TYPE_TO_CONFIG_PREFIX = {
    AgentType.MAIN: "",
    AgentType.SUB_AGENT: "SubAgent",
    AgentType.EMBEDDING: "Embedding",
}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        return json.load(file)


def get_model_config(
    config: dict[str, Any],
    provider: str,
    agent_type: AgentType,
) -> ModelConfig:
    agent_prefix = AGENT_TYPE_TO_CONFIG_PREFIX[agent_type]
    model_type = ModelType(provider)

    if model_type == ModelType.AZURE:
        _check_azure_config(config, agent_prefix)
        api_key = (
            config["azureApiKey"]
            if len(config["azureApiKey"]) > 0
            else os.environ["AZURE_OPENAI_API_KEY"]
        )
        model_name = config[f"azure{agent_prefix}DeploymentName"]
        if agent_type != AgentType.EMBEDDING:
            context_length = int(config[f"azure{agent_prefix}CtxLen"])
            tokenizer = get_encoding(encoding_name_for_model("gpt-3.5-turbo"))
    elif model_type == ModelType.LOCAL:
        _check_local_config(config, agent_prefix)
        api_key = "lm-studio"
        model_name = config[f"local{agent_prefix}ModelName"]
        context_length = int(
            config[f"local{agent_prefix}CtxLen"]
            if len(config[f"local{agent_prefix}CtxLen"]) > 0
            else DEFAULT_EMBEDDING_CONTEXT_LENGTH
        )
        if agent_type != AgentType.EMBEDDING:
            tokenizer = AutoTokenizer.from_pretrained(
                config[f"local{agent_prefix}TokenizerNameOrPath"],
                use_fast=True,
                token=config["hfToken"],
            )
    elif model_type == ModelType.OPENAI:
        _check_openai_config(config, agent_prefix)
        api_key = (
            config["openAIApiKey"]
            if len(config["openAIApiKey"]) > 0
            else os.environ["OPENAI_API_KEY"]
        )
        model_name = (
            config[f"openAI{agent_prefix}ModelName"]
            if agent_type != AgentType.EMBEDDING
            else DEFAULT_OPENAI_EMBEDDING_MODEL
        )
        if agent_type != AgentType.EMBEDDING:
            context_length = OPENAI_MODELS[model_name]
            tokenizer = get_encoding(encoding_name_for_model("gpt-3.5-turbo"))
    else:
        raise ValueError("Invalid model type")

    return ModelConfig(
        api_key=api_key,
        context_length=(
            context_length
            if agent_type != AgentType.EMBEDDING
            else DEFAULT_EMBEDDING_CONTEXT_LENGTH
        ),
        model_name=model_name,
        model_type=model_type,
        tokenizer=tokenizer if agent_type != AgentType.EMBEDDING else None,
        port=(
            int(config[f"local{agent_prefix}Port"])
            if model_type == ModelType.LOCAL
            and len(config[f"local{agent_prefix}Port"]) > 0
            else None
        ),
    )


def get_whisper_config(config: dict[str, Any], provider: str) -> WhisperConfig:
    whisper_provider = ModelType(provider)

    api_key = None
    port = None
    if whisper_provider == ModelType.OPENAI:
        if (
            not _is_valid_config_field(config, "openAIApiKey")
            and os.environ.get("OPENAI_API_KEY") is None
        ):
            raise ValueError("OpenAI API key for Whisper not found in config")

        api_key = (
            config.get("openAIApiKey")
            if len(config["openAIApiKey"]) > 0
            else os.environ.get("OPENAI_API_KEY")
        )
    elif whisper_provider == ModelType.LOCAL:
        if not _is_valid_config_field(config, "localWhisperPort"):
            raise ValueError("Local Whisper port not found in config")
        port = int(config["localWhisperPort"])
    else:
        raise ValueError("Invalid Whisper provider")

    return WhisperConfig(provider=whisper_provider, api_key=api_key, port=port)


def get_tiny_agent_config(config_path: str) -> TinyAgentConfig:
    config = load_config(config_path)

    if (provider := config.get("provider")) is None or len(provider) == 0:
        raise ValueError("Provider not found in config")

    if (sub_agent_provider := config.get("subAgentProvider")) is None or len(
        sub_agent_provider
    ) == 0:
        raise ValueError("Subagent provider not found in config")

    use_in_context_example_retriever = config.get("useToolRAG")
    if (
        use_in_context_example_retriever is False
        and config.get("toolRAGProvider") is None
    ):
        raise ValueError("In-context example retriever provider not found in config")

    whisper_provider = config.get("whisperProvider")
    if whisper_provider is None or len(whisper_provider) == 0:
        raise ValueError("Whisper provider not found in config")

    # Get the model configs
    llmcompiler_config = get_model_config(config, config["provider"], AgentType.MAIN)
    sub_agent_config = get_model_config(
        config,
        sub_agent_provider,
        AgentType.SUB_AGENT,
    )
    if use_in_context_example_retriever is True:
        embedding_model_config = get_model_config(
            config, config["toolRAGProvider"], AgentType.EMBEDDING
        )

    # Azure config
    azure_api_version = config.get("azureApiVersion")
    azure_endpoint = (
        config.get("azureEndpoint")
        if len(config["azureEndpoint"]) > 0
        else os.environ.get("AZURE_OPENAI_ENDPOINT")
    )

    # Get the other config values
    hf_token = config.get("hfToken")
    zoom_access_token = config.get("zoomAccessToken")

    # Get the whisper API key which is just the OpenAI API key
    if (
        not _is_valid_config_field(config, "openAIApiKey")
        and os.environ.get("OPENAI_API_KEY") is None
    ):
        raise ValueError("OpenAI API key is needed for whisper API.")

    whisper_config = get_whisper_config(config, whisper_provider)

    apps = set()
    for app in App:
        if config[f"{app.value}Enabled"]:
            apps.add(app)

    return TinyAgentConfig(
        apps=apps,
        custom_instructions=config.get("customInstructions"),
        llmcompiler_config=llmcompiler_config,
        sub_agent_config=sub_agent_config,
        embedding_model_config=(
            embedding_model_config if use_in_context_example_retriever else None
        ),
        azure_api_version=azure_api_version,
        azure_endpoint=azure_endpoint,
        hf_token=hf_token,
        zoom_access_token=zoom_access_token,
        whisper_config=whisper_config,
    )


def _is_valid_config_field(config: dict[str, Any], field: str) -> bool:
    return (field_value := config.get(field)) is not None and len(field_value) > 0


def _check_azure_config(config: dict[str, Any], agent_prefix: str) -> None:
    if (
        not _is_valid_config_field(config, "azureApiKey")
        and os.environ.get("AZURE_OPENAI_API_KEY") is None
    ):
        raise ValueError("Azure API key not found in config")
    if not _is_valid_config_field(config, f"azure{agent_prefix}DeploymentName"):
        raise ValueError(
            f"Azure {agent_prefix} model deployment name not found in config"
        )
    if agent_prefix != AGENT_TYPE_TO_CONFIG_PREFIX[
        AgentType.EMBEDDING
    ] and not _is_valid_config_field(config, f"azure{agent_prefix}CtxLen"):
        raise ValueError(f"Azure {agent_prefix} context length not found in config")
    if not _is_valid_config_field(config, "azureApiVersion"):
        raise ValueError("Azure API version not found in config")
    if not _is_valid_config_field(config, "azureEndpoint"):
        raise ValueError("Azure endpoint not found in config")


def _check_openai_config(config: dict[str, Any], agent_prefix: str) -> None:
    if (
        not _is_valid_config_field(config, "openAIApiKey")
        and os.environ.get("OPENAI_API_KEY") is None
    ):
        raise ValueError("OpenAI API key not found in config")
    # The embedding model for OpenAI API only supports "text-embedding-3-small" hence
    # we don't need to check for the model name for the embedding model
    if agent_prefix != AGENT_TYPE_TO_CONFIG_PREFIX[
        AgentType.EMBEDDING
    ] and not _is_valid_config_field(config, f"openAI{agent_prefix}ModelName"):
        raise ValueError(f"OpenAI {agent_prefix} model name not found in config")


def _check_local_config(config: dict[str, Any], agent_prefix: str) -> None:
    # TinyAgent does not support local embedding models. Hence, we only need to check for
    # the context length, port, and tokenizer name or path for the local planner model.
    is_embedding_model = (
        agent_prefix == AGENT_TYPE_TO_CONFIG_PREFIX[AgentType.EMBEDDING]
    )
    if not is_embedding_model and not _is_valid_config_field(
        config, f"local{agent_prefix}CtxLen"
    ):
        raise ValueError(f"Local {agent_prefix} context length not found in config")
    if not is_embedding_model and not _is_valid_config_field(
        config, f"local{agent_prefix}Port"
    ):
        raise ValueError(f"Local {agent_prefix} port not found in config")
    if not is_embedding_model and not _is_valid_config_field(
        config, f"local{agent_prefix}TokenizerNameOrPath"
    ):
        raise ValueError(
            f"Local {agent_prefix} tokenizer name or path not found in config"
        )
    if is_embedding_model and not _is_valid_config_field(
        config, f"local{agent_prefix}ModelName"
    ):
        raise ValueError(f"Local {agent_prefix} model name not found in config")
