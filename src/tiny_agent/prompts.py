import datetime
from typing import Sequence

from src.llm_compiler.constants import (
    END_OF_PLAN,
    JOINNER_FINISH,
    JOINNER_REPLAN,
    SUMMARY_RESULT,
)
from src.tiny_agent.models import TinyAgentToolName
from src.tools.base import StructuredTool, Tool
from src.utils.logger_utils import log

NOW = datetime.datetime.now()


DEFAULT_PLANNER_IN_CONTEXT_EXAMPLES_PROMPT = (
    "Question: Notify Lutfi Eren Erdogan about the upcoming Apple meeting that is going to start at 3PM on Friday.\n"
    '1. get_phone_number("Lutfi Eren Erdogan")\n'
    '2. send_sms(["$1"], "Hey Lutfi, just wanted to let you know about the upcoming Apple meeting. It\'s going to be at 3 PM on Friday.")\n'
    "Thought: I have succesfully found the contact and sent the message.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "Question: Create a zoom meeting for the upcoming Apple meeting with Eren Erdoğan.\n"
    '1. get_email_address("Eren Erdoğan")\n'
    '2. get_zoom_meeting_link("Apple Meeting", "2022-10-14 15:00:00", 60, ["$1"])\n'
    '3. create_calendar_event("Apple Meeting", "2022-10-14 15:00:00", "2022-10-14 16:00:00", "$2", [], "", None)\n'
    "Thought: I have succesfully created the calendar event.\n"
    f"4. join(){END_OF_PLAN}\n"
    "###\n"
    "Question: Show directions to Apple Park.\n"
    '1. maps_show_directions("", "Apple Park", "d")\n'
    "Thought: I have succesfully shown the directions.\n"
    f"2. join(){END_OF_PLAN}\n"
    "###\n"
    "Question: Send an email to Amir saying that the meeting is postponed to next week.\n"
    '1. get_email_address("Amir")\n'
    '2. compose_new_email(["$1"], [], "Meeting Postponed," "", [])\n'
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
)


TOOL_SPECIFIC_PROMPTS: list[tuple[set[TinyAgentToolName], str]] = [
    (
        {
            TinyAgentToolName.GET_EMAIL_ADDRESS,
            TinyAgentToolName.COMPOSE_NEW_EMAIL,
            TinyAgentToolName.REPLY_TO_EMAIL,
            TinyAgentToolName.FORWARD_EMAIL,
        },
        f" - Before sending an email, you MUST use the {TinyAgentToolName.GET_EMAIL_ADDRESS.value} tool to get the email addresses of the recipients and cc, unless you are explicitly given their email addresses.\n",
    ),
    (
        {TinyAgentToolName.GET_PHONE_NUMBER, TinyAgentToolName.SEND_SMS},
        f" - Before sending an SMS, you MUST use the {TinyAgentToolName.GET_PHONE_NUMBER.value} tool to get the phone number of the contact, unless you are explicitly given their phone number.\n"
        " - If you need to send an SMS message to multiple contacts, send it in one message, unless specified otherwise.\n",
    ),
    (
        {
            TinyAgentToolName.GET_EMAIL_ADDRESS,
            TinyAgentToolName.COMPOSE_NEW_EMAIL,
            TinyAgentToolName.REPLY_TO_EMAIL,
            TinyAgentToolName.FORWARD_EMAIL,
            TinyAgentToolName.GET_PHONE_NUMBER,
            TinyAgentToolName.SEND_SMS,
        },
        f" - If you need to send an email or an sms using {TinyAgentToolName.COMPOSE_NEW_EMAIL.value}, {TinyAgentToolName.REPLY_TO_EMAIL.value}, {TinyAgentToolName.FORWARD_EMAIL.value}, or {TinyAgentToolName.SEND_SMS.value} tools, "
        f"you MUST send it before calling join(), or you WILL BE PENALIZED!\n",
    ),
    (
        {TinyAgentToolName.GET_ZOOM_MEETING_LINK},
        f" - If you need to create a zoom meeting, you MUST use {TinyAgentToolName.GET_ZOOM_MEETING_LINK.value} to get the newly created zoom meeting link.\n",
    ),
    (
        {TinyAgentToolName.OPEN_NOTE, TinyAgentToolName.APPEND_NOTE_CONTENT},
        f" - If you need to append some content to a note, you DON'T HAVE TO call {TinyAgentToolName.OPEN_NOTE.value} before calling {TinyAgentToolName.APPEND_NOTE_CONTENT.value}. You can directly use {TinyAgentToolName.APPEND_NOTE_CONTENT.value} to append some content to the specific note.\n",
    ),
    (
        {TinyAgentToolName.MAPS_OPEN_LOCATION, TinyAgentToolName.MAPS_SHOW_DIRECTIONS},
        f" - If you need to show directions to a place, you DON'T HAVE TO call {TinyAgentToolName.MAPS_OPEN_LOCATION.value} before calling {TinyAgentToolName.MAPS_SHOW_DIRECTIONS.value}. You can directly use {TinyAgentToolName.MAPS_SHOW_DIRECTIONS.value} to show directions to the specific place.\n",
    ),
]


def get_planner_custom_instructions_prompt(
    tools: Sequence[Tool | StructuredTool], custom_instructions: str | None
) -> str:
    prompt = []
    prompt.append(
        " - You need to start your plan with the '1.' call\n"
        f" - Today's date is {NOW.strftime('%A %Y-%m-%d %H:%M')}\n"
        " - Unless otherwise specified, the default meeting duration is 60 minutes.\n"
        " - Do not use named arguments in your tool calls.\n"
        " - You MUST end your plans with the 'join()' call and a '\\n' character.\n"
        " - You MUST fill every argument in the tool calls, even if they are optional.\n"
        " - The format for dates MUST be in ISO format of 'YYYY-MM-DD HH:MM:SS', unless other specified.\n"
        " - If you want to use the result of a previous tool call, you MUST use the '$' sign followed by the index of the tool call.\n"
        f" - You MUST ONLY USE join() at the very very end of the plan, or you WILL BE PENALIZED.\n"
    )

    for tool_set, instructions in TOOL_SPECIFIC_PROMPTS:
        if any(TinyAgentToolName(tool.name) in tool_set for tool in tools):
            prompt += instructions

    if custom_instructions is not None and len(custom_instructions) > 0:
        prompt.append(f" - {custom_instructions}")

    return "".join(prompt)


PLANNER_PROMPT_REPLAN = (
    "Question: Say hi to Sid via SMS.\n\n"
    "Previous Plan:\n\n"
    "1. join()\n"
    "Observation:\nThe plan generation was stopped due to an error in tool 1. get_contact_info('Sid')! "
    "Error: Tool get_contact_info not found. You MUST correct this error and try again!"
    "\n"
    "Current Plan:\n\n"
    f"Thought: The error is fixable since I have the {TinyAgentToolName.GET_PHONE_NUMBER.value} tool to retrieve the phone number of Sid. Then I will proceed with sending the SMS.\n"
    '1. get_phone_number("Sid")\n'
    '2. send_sms("$2", "Hi Sid!")\n'
    "Thought: I have succesfully created the retrieved the phone number and sent the SMS.\n"
    f"4. join(){END_OF_PLAN}\n"
    "###\n"
    "Question: Summarize 'Apple Demo.pdf'.\n\n"
    "Previous Plan:\n\n"
    '1. open_and_get_file_path("Apple Demo")\n'
    "2. join()\n"
    "Observation: summarize_pdf() takes 1 positional arguments but 2 were given! You MUST correct this error and try again!"
    "\n"
    "Current Plan:\n\n"
    f"Thought: Previous plan tried to call the summarize_pdf() tool with the wrong number of arguments. I will correct this and try again.\n"
    '1. open_and_get_file_path("Apple Demo")\n'
    '2. summarize_pdf("$1")\n'
    "Thought: I have succesfully opened the file and summarized it.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
)

JOINNER_REPLAN_RULES = (
    f" - If you think the plan is not completed yet or an error in the plan is fixable, you should output {JOINNER_REPLAN}.\n"
    f" - If the plan is fixable, you will see a message like 'try again'. If you don't see this message, the error is NOT fixable and you MUST output an error message using 'Action: {JOINNER_FINISH}(<your error message>)'\n"
)

JOINNER_FINISH_RULES = (
    f" - If you need to answer some knowledge question, just answer it directly using 'Action: {JOINNER_FINISH}(<your answer>)'.\n"
    f" - If you need to return the result of a summary (summarize_pdf), you MUST use 'Action: {JOINNER_FINISH}({SUMMARY_RESULT})'\n"
    f" - If there is an error in one of the tool calls and it is not fixable, you should provide a user-friendly error message using 'Action: {JOINNER_FINISH}(<your error message>)'.\n"
)

REPLAN_EXAMPLES = [
    "Question: Say hi to Sid via SMS.\n"
    "join()\n"
    "Observation: The plan generation was stopped due to an error in tool 1. get_contact_info('Sid')! "
    "Error: Tool get_contact_info not found. You MUST correct this error and try again!"
    "Thought: The error is fixable so I need to replan and try again.\n"
    f"Action: {JOINNER_REPLAN}\n"
]

FINISH_EXAMPLES = [
    "Question: Create a zoom meeting for the upcoming Apple meeting with Eren Erdoğan. \n"
    'get_email_address("Eren Erdoğan")\n'
    "Observation: eren@gmail.com\n"
    'get_zoom_meeting_link("Apple Meeting", "2022-10-14 15:00:00", 60, ["$1"])\n'
    "Observation: https://zoom.us/j/1234567890?pwd=abc123\n"
    'create_calendar_event("Apple Meeting", "2022-10-14 15:00:00", "2022-10-14 16:00:00", "Apple HQ", "$2", None)\n'
    "Observation: Event created successfully\n"
    "Thought: I don't need to answer a question.\n"
    f"Action: {JOINNER_FINISH}(Task completed!)\n",
    "Question: What is the content of the Apple meeting notes? \n"
    'get_note_content("Apple Meeting")\n'
    "Observation: The meeting is about the new iPhone release.\n"
    "Thought: I can just answer the question directly.\n"
    f"Action: {JOINNER_FINISH}(The meeting is about the new iPhone release.)\n",
    "Question: Compose a new email to John, attaching the Project.pdf file.\n"
    'get_email_address("John")\n'
    "Observation: john@doe.com"
    'open_and_get_file_path("Project")\n'
    "Observation: /Users/eren/Downloads/Project.pdf\n"
    'compose_new_email([john@doe.com], [], "Project Update", "Please find the attached project update.", ["/Users/eren/Downloads/Project.pdf"])\n'
    "Observation: There was an error while composing the email.\n"
    "Thought: There was an error with the compose_new_email tool call and it is not possible to fix it. I need to provide a user-friendly error message.\n"
    f"Action: {JOINNER_FINISH}(There was an error while composing the email. Please try again later.)\n",
    "Question: Summarize the Apple Demo file. \n"
    "open_and_get_file_path(Apple Demo)\n"
    "Observation: /Users/eren/Downloads/Apple_Demo.pdf\n"
    "summarize_pdf(/Users/eren/Downloads/Apple_Demo.pdf)\n"
    "Observation: The new iPhone is going to be released in 2023.\n"
    f"Action: {JOINNER_FINISH}({SUMMARY_RESULT})\n",
]

OUTPUT_PROMPT = (
    "Follow these rules:\n"
    f" - You MUST only output either {JOINNER_FINISH} or {JOINNER_REPLAN}, or you WILL BE PENALIZED.\n"
    f"{JOINNER_FINISH_RULES}"
    f"{JOINNER_REPLAN_RULES}"
    "\n"
    "Here are some examples:\n"
    + "###\n".join(FINISH_EXAMPLES)
    + "###\n"
    + "###\n".join(REPLAN_EXAMPLES)
    + "###\n"
)


OUTPUT_PROMPT_FINAL = (
    "Follow these rules:\n"
    f" - You MUST only output {JOINNER_FINISH}, or you WILL BE PENALIZED.\n"
    f"{JOINNER_FINISH_RULES}"
    "\n"
    "Here are some examples:\n" + "###\n".join(FINISH_EXAMPLES) + "###\n"
)
