from enum import Enum

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from src.tiny_agent.models import NotesMode
from src.tiny_agent.sub_agents.sub_agent import SubAgent


class NotesAgent(SubAgent):

    async def __call__(
        self,
        name: str,
        content: str,
        prev_content: str = "",
        mode: NotesMode = NotesMode.NEW,
    ) -> str:
        # Construct the prompt
        if mode == NotesMode.NEW:
            notes_llm_system_prompt = (
                "You are an expert note taking agent. Given the plain text content, you MUST generate a compelling and "
                "formatted HTML version of the note (with <html>, <head>, <body> tags, etc.) The content of the note should be rich, well-structured, and verbose. "
            )
        elif mode == NotesMode.APPEND:
            notes_llm_system_prompt = (
                "You are an expert note-taking agent that specializes in appending new content to existing notes. "
                "Given the content of an existing note and the content to append, you MUST generate a continuation of the note that "
                "seamlessly integrates with the existing content. Your additions should maintain the tone, style, "
                "and subject matter of the original note. You MUST ONLY output the appended content in HTML format, DO NOT include the entire note, or you WILL BE PENALIZED.\n"
                f"Previous Content:\n{prev_content}\n"
            )

        notes_llm_system_prompt += (
            "The note should include appropriate use of headings, bold and italic text for emphasis, "
            "bullet points for lists, and paragraph tags for separation of ideas. DO NOT include the '<!DOCTYPE html>' tag."
        )

        # Add custom instructions to the system prompt if specified
        if self._custom_instructions is not None:
            notes_llm_system_prompt += (
                "\nHere are some general facts about the user's preferences, "
                f"you MUST keep these in mind when generating your note:\n{self._custom_instructions}"
            )

        notes_human_prompt = (
            "Note Title: {name}\nNew Text Content: {content}\nHTML Content:\n"
        )
        messages = [
            SystemMessage(content=notes_llm_system_prompt),
            HumanMessage(content=notes_human_prompt.format(name=name, content=content)),
        ]
        plain_text_content = self.check_context_length(messages, content)

        # If the plain text content is too long, then get the first X tokens of the subagent llm
        if plain_text_content is not None:
            messages = [
                SystemMessage(content=notes_llm_system_prompt),
                HumanMessage(
                    content=notes_human_prompt.format(name=name, content=content)
                ),
            ]

        # Generate the HTML content for the note
        html_content = await self._llm.apredict_messages(messages)

        # If the html doesn't start with a <html> tag, then add it
        soup = BeautifulSoup(str(html_content.content), "html.parser")
        if not soup.find("html"):
            soup = BeautifulSoup(
                f"<html>{str(html_content.content)}</html>", "html.parser"
            )

        return str(soup.find("html"))
