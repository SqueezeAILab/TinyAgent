import re
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

from src.tiny_agent.models import ComposeEmailMode
from src.tiny_agent.sub_agents.sub_agent import SubAgent


class ComposeEmailAgent(SubAgent):
    _query: str

    @property
    def query(self) -> str:
        return self._query

    @query.setter
    def query(self, query: str) -> None:
        self._query = query

    async def __call__(
        self,
        context: str,
        email_thread: str = "",
        mode: ComposeEmailMode = ComposeEmailMode.NEW,
    ) -> str:
        # Define the system prompt for the LLM to generate HTML content
        context = context.strip()
        cleaned_thread = re.sub(r"\n+", "\n", email_thread.strip())
        if mode == ComposeEmailMode.NEW:
            email_llm_system_prompt = (
                "You are an expert email composer agent. Given an email content or a user query, you MUST generate a well-formatted and "
                "informative email. The email should include a polite greeting, a detailed body, and a "
                "professional sign-off. You MUST NOT include a subject. The email should be well-structured and free of grammatical errors."
            )
        elif mode == ComposeEmailMode.REPLY:
            email_llm_system_prompt = (
                "You are an expert email composer agent. Given the content of the past email thread and a user query, "
                "you MUST generate a well-formatted and informative reply to the last email in the thread. "
                "The email should include a polite greeting, a detailed body, and a "
                "professional sign-off. You MUST NOT include a subject. The email should be well-structured and free of grammatical errors."
            )
            context += f"\nEmail Thread:\n{cleaned_thread}"
        elif mode == ComposeEmailMode.FORWARD:
            email_llm_system_prompt = (
                "You are an expert email composer agent. Given the content of the past email thread and a user query, "
                "you MUST generate a very concise and informative forward of the last email in the thread. "
            )
            context += f"\nEmail Thread:\n{cleaned_thread}"

        # Add custom instructions to the system prompt if specified
        if self._custom_instructions is not None:
            email_llm_system_prompt += (
                "\nHere are some general facts about the user's preferences, "
                f"you MUST keep these in mind when writing your email:\n{self._custom_instructions}"
            )

        email_human_prompt = "Context:\n{context}\nQuery: {query}\nEmail Body:\n"
        messages = [
            SystemMessage(content=email_llm_system_prompt),
            HumanMessage(
                content=email_human_prompt.format(context=context, query=self._query)
            ),
        ]

        # If the context content is too long, then get the first X tokens of the subagent llm
        new_context = self.check_context_length(messages, context)
        if new_context is not None:
            messages = [
                SystemMessage(content=email_llm_system_prompt),
                HumanMessage(
                    content=email_human_prompt.format(
                        context=new_context, query=self._query
                    )
                ),
            ]

        # Generate the HTML content for the email
        email_content = await self._llm.apredict_messages(messages)

        return str(email_content.content)
