from src.llm_compiler.constants import END_OF_PLAN, SUMMARY_RESULT
from src.llm_compiler.llm_compiler import LLMCompiler
from src.llm_compiler.planner import generate_llm_compiler_prompt
from src.tiny_agent.computer import Computer
from src.tiny_agent.config import TinyAgentConfig
from src.tiny_agent.prompts import (
    DEFAULT_PLANNER_IN_CONTEXT_EXAMPLES_PROMPT,
    OUTPUT_PROMPT,
    OUTPUT_PROMPT_FINAL,
    PLANNER_PROMPT_REPLAN,
    get_planner_custom_instructions_prompt,
)
from src.tiny_agent.sub_agents.compose_email_agent import ComposeEmailAgent
from src.tiny_agent.sub_agents.notes_agent import NotesAgent
from src.tiny_agent.sub_agents.pdf_summarizer_agent import PDFSummarizerAgent
from src.tiny_agent.tiny_agent_tools import (
    get_tiny_agent_tools,
    get_tool_names_from_apps,
)
from src.tiny_agent.tool_rag.base_tool_rag import BaseToolRAG
from src.tiny_agent.tool_rag.classifier_tool_rag import ClassifierToolRAG
from src.utils.model_utils import get_embedding_model, get_model


class TinyAgent:
    _DEFAULT_TOP_K = 6

    config: TinyAgentConfig
    agent: LLMCompiler
    computer: Computer
    notes_agent: NotesAgent
    pdf_summarizer_agent: PDFSummarizerAgent
    compose_email_agent: ComposeEmailAgent
    tool_rag: BaseToolRAG

    def __init__(self, config: TinyAgentConfig) -> None:
        self.config = config

        # Define the models
        llm = get_model(
            model_type=config.llmcompiler_config.model_type.value,
            model_name=config.llmcompiler_config.model_name,
            api_key=config.llmcompiler_config.api_key,
            stream=False,
            vllm_port=config.llmcompiler_config.port,
            temperature=0,
            azure_api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
            azure_deployment=config.llmcompiler_config.model_name,
        )
        planner_llm = get_model(
            model_type=config.llmcompiler_config.model_type.value,
            model_name=config.llmcompiler_config.model_name,
            api_key=config.llmcompiler_config.api_key,
            stream=True,
            vllm_port=config.llmcompiler_config.port,
            temperature=0,
            azure_api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
            azure_deployment=config.llmcompiler_config.model_name,
        )
        sub_agent_llm = get_model(
            model_type=config.sub_agent_config.model_type.value,
            model_name=config.sub_agent_config.model_name,
            api_key=config.sub_agent_config.api_key,
            stream=False,
            vllm_port=config.sub_agent_config.port,
            temperature=0,
            azure_api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
            azure_deployment=config.sub_agent_config.model_name,
        )

        self.computer = Computer()
        self.notes_agent = NotesAgent(
            sub_agent_llm, config.sub_agent_config, config.custom_instructions
        )
        self.pdf_summarizer_agent = PDFSummarizerAgent(
            sub_agent_llm, config.sub_agent_config, config.custom_instructions
        )
        self.compose_email_agent = ComposeEmailAgent(
            sub_agent_llm, config.sub_agent_config, config.custom_instructions
        )

        tools = get_tiny_agent_tools(
            computer=self.computer,
            notes_agent=self.notes_agent,
            pdf_summarizer_agent=self.pdf_summarizer_agent,
            compose_email_agent=self.compose_email_agent,
            tool_names=get_tool_names_from_apps(config.apps),
            zoom_access_token=config.zoom_access_token,
        )

        # Define LLMCompiler
        self.agent = LLMCompiler(
            tools=tools,
            planner_llm=planner_llm,
            planner_custom_instructions_prompt=get_planner_custom_instructions_prompt(
                tools=tools, custom_instructions=config.custom_instructions
            ),
            planner_example_prompt=DEFAULT_PLANNER_IN_CONTEXT_EXAMPLES_PROMPT,
            planner_example_prompt_replan=PLANNER_PROMPT_REPLAN,
            planner_stop=[END_OF_PLAN],
            planner_stream=True,
            agent_llm=llm,
            joinner_prompt=OUTPUT_PROMPT,
            joinner_prompt_final=OUTPUT_PROMPT_FINAL,
            max_replans=2,
            benchmark=False,
        )

        # Define ToolRAG
        if config.embedding_model_config is not None:
            embedding_model = get_embedding_model(
                model_type=config.embedding_model_config.model_type.value,
                model_name=config.embedding_model_config.model_name,
                api_key=config.embedding_model_config.api_key,
                azure_endpoint=config.azure_endpoint,
                azure_embedding_deployment=config.embedding_model_config.model_name,
                azure_api_version=config.azure_api_version,
                local_port=config.embedding_model_config.port,
                context_length=config.embedding_model_config.context_length,
            )
            self.tool_rag = ClassifierToolRAG(
                embedding_model=embedding_model,
                tools=tools,
            )

    async def arun(self, query: str) -> str:
        if self.config.embedding_model_config is not None:
            tool_rag_results = self.tool_rag.retrieve_examples_and_tools(
                query, top_k=TinyAgent._DEFAULT_TOP_K
            )

            new_tools = get_tiny_agent_tools(
                computer=self.computer,
                notes_agent=self.notes_agent,
                pdf_summarizer_agent=self.pdf_summarizer_agent,
                compose_email_agent=self.compose_email_agent,
                tool_names=tool_rag_results.retrieved_tools_set,
                zoom_access_token=self.config.zoom_access_token,
            )

            self.agent.planner.system_prompt = generate_llm_compiler_prompt(
                tools=new_tools,
                example_prompt=tool_rag_results.in_context_examples_prompt,
                custom_instructions=get_planner_custom_instructions_prompt(
                    tools=new_tools, custom_instructions=self.config.custom_instructions
                ),
            )

        self.compose_email_agent.query = query
        result = await self.agent.arun(query)

        if result == SUMMARY_RESULT:
            result = self.pdf_summarizer_agent.cached_summary_result

        return result
