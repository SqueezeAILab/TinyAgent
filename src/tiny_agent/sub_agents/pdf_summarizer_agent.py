import fitz
from langchain_core.messages import HumanMessage, SystemMessage

from src.tiny_agent.sub_agents.sub_agent import SubAgent

CONTEXT_LENGTHS = {"gpt-4-1106-preview": 127000, "gpt-3.5-turbo": 16000}


class PDFSummarizerAgent(SubAgent):
    _cached_summary_result: str

    @property
    def cached_summary_result(self) -> str:
        return self._cached_summary_result

    async def __call__(self, pdf_path: str) -> str:
        # Check if the file exists
        if (
            pdf_path is None
            or len(pdf_path) <= 0
            or pdf_path
            in (
                "No file found after fuzzy matching.",
                "No file found with exact or fuzzy name.",
            )
        ):
            return "The PDF file path is invalid or the file doesn't exist."

        try:
            pdf_content = PDFSummarizerAgent._extract_text_from_pdf(pdf_path)
        except Exception as e:
            return f"An error occurred while extracting the content from the PDF file: {str(e)}"

        if len(pdf_content) <= 0:
            return "The PDF file is empty or the content couldn't be extracted."

        # Construct the prompt
        pdf_summarizer_llm_system_prompt = (
            "You are an expert PDF summarizer agent. Given the PDF content, you MUST generate an informative and verbose "
            "summary of the content. The summary should include the main points and key details of the content. "
        )
        pdf_summarizer_human_prompt = "PDF Content:\n{pdf_content}\nSummary:\n"
        messages = [
            SystemMessage(content=pdf_summarizer_llm_system_prompt),
            HumanMessage(
                content=pdf_summarizer_human_prompt.format(pdf_content=pdf_content)
            ),
        ]

        # If the PDF content is too long, then get the first X tokens of the subagent llm
        pdf_content = self.check_context_length(messages, pdf_content)
        if pdf_content is not None:
            messages = [
                SystemMessage(content=pdf_summarizer_llm_system_prompt),
                HumanMessage(
                    content=pdf_summarizer_human_prompt.format(pdf_content=pdf_content)
                ),
            ]

        # Call LLM
        summary = await self._llm.apredict_messages(messages)

        # Cache the summary result
        self._cached_summary_result = str(summary.content)

        return self._cached_summary_result

    @staticmethod
    def _extract_text_from_pdf(pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            text.append(page.get_text())  # type: ignore
        doc.close()
        return "".join(text).replace("\n", " ")
