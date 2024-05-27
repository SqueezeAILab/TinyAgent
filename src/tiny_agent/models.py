import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Collection

import torch
from tiktoken import Encoding
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

streaming_queue = asyncio.Queue[str | None]()

LLM_ERROR_TOKEN = "###LLM_ERROR_TOKEN###"

TINY_AGENT_DIR = os.path.expanduser("~/Library/Application Support/TinyAgent")

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast | Encoding


class ModelType(Enum):
    AZURE = "azure"
    OPENAI = "openai"
    LOCAL = "local"


class App(Enum):
    CALENDAR = "calendar"
    CONTACTS = "contacts"
    FILES = "files"
    MAIL = "mail"
    MAPS = "maps"
    NOTES = "notes"
    REMINDERS = "reminders"
    SMS = "sms"
    ZOOM = "zoom"


class AgentType(Enum):
    MAIN = "main"
    SUB_AGENT = "sub_agent"
    EMBEDDING = "embedding"


@dataclass
class ModelConfig:
    api_key: str
    context_length: int
    model_name: str
    model_type: ModelType
    tokenizer: Tokenizer | None
    port: int | None


@dataclass
class WhisperConfig:
    # Azure is not yet supported for whisper
    provider: ModelType
    api_key: str | None
    port: int | None


@dataclass
class TinyAgentConfig:
    # Custom configs
    apps: Collection[App]
    custom_instructions: str | None
    # Config for the LLMCompiler model
    llmcompiler_config: ModelConfig
    # Config for the sub-agent LLM model
    sub_agent_config: ModelConfig
    # Config for the embedding model
    embedding_model_config: ModelConfig | None
    # Azure model config
    azure_api_version: str | None
    azure_endpoint: str | None
    # Other tokens
    hf_token: str | None
    zoom_access_token: str | None
    # Whisper config
    whisper_config: WhisperConfig


class TinyAgentToolName(Enum):
    GET_PHONE_NUMBER = "get_phone_number"
    GET_EMAIL_ADDRESS = "get_email_address"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    OPEN_AND_GET_FILE_PATH = "open_and_get_file_path"
    SUMMARIZE_PDF = "summarize_pdf"
    COMPOSE_NEW_EMAIL = "compose_new_email"
    REPLY_TO_EMAIL = "reply_to_email"
    FORWARD_EMAIL = "forward_email"
    MAPS_OPEN_LOCATION = "maps_open_location"
    MAPS_SHOW_DIRECTIONS = "maps_show_directions"
    CREATE_NOTE = "create_note"
    OPEN_NOTE = "open_note"
    APPEND_NOTE_CONTENT = "append_note_content"
    CREATE_REMINDER = "create_reminder"
    SEND_SMS = "send_sms"
    GET_ZOOM_MEETING_LINK = "get_zoom_meeting_link"


@dataclass
class InContextExample:
    example: str
    embedding: torch.Tensor
    tools: list[TinyAgentToolName]


class ComposeEmailMode(Enum):
    NEW = "new"
    REPLY = "reply"
    FORWARD = "forward"


class NotesMode(Enum):
    NEW = "new"
    APPEND = "append"


class TransportationOptions(Enum):
    DRIVING = "d"
    WALKING = "w"
    PUBLIC_TRANSIT = "r"
