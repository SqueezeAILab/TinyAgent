import datetime
import os
import re
from typing import Any, Collection, Sequence

from src.tiny_agent.computer import Computer
from src.tiny_agent.config import App
from src.tiny_agent.models import (
    ComposeEmailMode,
    NotesMode,
    TinyAgentToolName,
    TransportationOptions,
)
from src.tiny_agent.sub_agents.compose_email_agent import ComposeEmailAgent
from src.tiny_agent.sub_agents.notes_agent import NotesAgent
from src.tiny_agent.sub_agents.pdf_summarizer_agent import PDFSummarizerAgent
from src.tiny_agent.tools.zoom import Zoom
from src.tools.base import StructuredTool, Tool


def get_datetime(date: str | None) -> datetime.datetime | None:
    if date is None or len(date) <= 0:
        return None
    try:
        return datetime.datetime.fromisoformat(date)
    except ValueError:
        return None


def ensure_phone_number_formatting(
    phone_numbers: Sequence[str], computer: Computer
) -> list[str]:
    pattern = r"\d+"
    return [
        (
            phone_number
            if re.search(pattern, phone_number) is not None
            else computer.contacts.get_phone_number(contact_name=phone_number)
        )
        for phone_number in phone_numbers
    ]


def ensure_email_formatting(
    email_addresses: Sequence[str], computer: Computer
) -> list[str]:
    return [
        email if "@" in email else computer.contacts.get_email_address(email)
        for email in email_addresses
    ]


def ensure_file_paths(file_paths: Sequence[str], computer: Computer) -> list[str]:
    return [
        (
            file_path
            if os.path.exists(file_path)
            else computer.spotlight_search.open(file_path)
        )
        for file_path in file_paths
    ]


def get_phone_number_tool(computer: Computer) -> Tool:
    async def get_phone_number(name: str) -> str:
        phone_number = computer.contacts.get_phone_number(contact_name=name)

        if phone_number == "No contacts found" or phone_number.startswith(
            "A contact for"
        ):
            return phone_number
        if len(phone_number) <= 0:
            return "Contact doesn't appear to have a phone number."

        # if the phone number doesn't start with a country code, then default to '+1'
        if not phone_number.startswith("+"):
            phone_number = f"+1{phone_number}"

        return phone_number.strip()

    return Tool(
        name=TinyAgentToolName.GET_PHONE_NUMBER.value,
        func=get_phone_number,
        description=(
            f"{TinyAgentToolName.GET_PHONE_NUMBER.value}(name: str) -> str\n"
            " - Search for a contact by name.\n"
            " - Returns the phone number of the contact.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.GET_PHONE_NUMBER.value}({args[0]})",
    )


def get_email_address_tool(computer: Computer) -> Tool:
    async def get_email_address(name: str) -> str:
        email_address = computer.contacts.get_email_address(contact_name=name)

        if email_address == "No contacts found" or email_address.startswith(
            "A contact for"
        ):
            return email_address
        if len(email_address) <= 0:
            return "Contact doesn't appear to have an email address."

        return email_address.strip()

    return Tool(
        name=TinyAgentToolName.GET_EMAIL_ADDRESS.value,
        func=get_email_address,
        description=(
            f"{TinyAgentToolName.GET_EMAIL_ADDRESS.value}(name: str) -> str\n"
            " - Search for a contact by name.\n"
            " - Returns the email address of the contact.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.GET_EMAIL_ADDRESS.value}({args[0]})",
    )


def get_create_calendar_event_tool(computer: Computer) -> Tool:
    async def create_calendar_event(
        title: str,
        start_date: str,
        end_date: str,
        location: str,
        invitees: list[str],
        notes: str,
        calendar: str,
    ) -> str:
        # Check whether the start_date and end_date are provided and are in correct format
        # If not, default to the current time and 1 hour later
        start_date_args = get_datetime(start_date)
        end_date_args = get_datetime(end_date)
        if start_date_args is None and end_date_args is None:
            start_date_args = datetime.datetime.now()
            end_date_args = start_date_args + datetime.timedelta(hours=1)
        elif start_date_args is None:
            end_date_args = datetime.datetime.fromisoformat(end_date)
            start_date_args = end_date_args - datetime.timedelta(hours=1)
        elif end_date_args is None:
            start_date_args = datetime.datetime.fromisoformat(start_date)
            end_date_args = start_date_args + datetime.timedelta(hours=1)

        args = {
            "title": title,
            "start_date": start_date_args,
            "end_date": end_date_args,
        }
        if location is not None and len(location) > 0:
            args["location"] = location
        if notes is not None and len(notes) > 0:
            args["notes"] = notes
        if calendar is not None and "None" not in calendar and len(calendar) > 0:
            args["calendar"] = calendar

        # Ensure consistent formatting of invitees and ensure they are email addresses
        if isinstance(invitees, str):
            args["invitees"] = ensure_email_formatting([invitees], computer)
        else:
            args["invitees"] = ensure_email_formatting(invitees, computer)

        return computer.calendar.create_event(**args)

    return Tool(
        name=TinyAgentToolName.CREATE_CALENDAR_EVENT.value,
        func=create_calendar_event,
        description=(
            f"{TinyAgentToolName.CREATE_CALENDAR_EVENT.value}("
            "title: str, "
            "start_date: str, "
            "end_date: str, "
            "location: str, "
            "invitees: list[str], "
            "notes: str, "
            "calendar: str"
            ") -> str\n"
            " - Create a calendar event.\n"
            " - The format for start_date and end_date is 'YYYY-MM-DD HH:MM:SS'.\n"
            " - For invitees, you need a list of email addresses; use an empty list if not applicable.\n"
            " - For location, notes, and calendar, use an empty string or None if not applicable.\n"
            " - Returns the status of the event creation.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.CREATE_CALENDAR_EVENT.value}("
            f"title={args[0]}, "
            f"start_date={args[1]}, "
            f"end_date={args[2]}, "
            f"location={args[3]}, "
            f"invitees={args[4]}, "
            f"notes={args[5]}, "
            f"calendar={args[6]})"
        ),
    )


def get_open_and_get_file_path_tool(computer: Computer) -> Tool:
    async def open_and_get_file_path(file_name: str) -> str:
        return computer.spotlight_search.open(file_name)

    return Tool(
        name=TinyAgentToolName.OPEN_AND_GET_FILE_PATH.value,
        func=open_and_get_file_path,
        description=(
            f"{TinyAgentToolName.OPEN_AND_GET_FILE_PATH.value}(file_name: str) -> str\n"
            " - Opens the file and returns its path.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.OPEN_AND_GET_FILE_PATH.value}({args[0]})",
    )


def get_summarize_pdf_tool(
    computer: Computer, pdf_summarizer_agent: PDFSummarizerAgent
) -> Tool:
    async def summarize_pdf(pdf_path: str) -> str:
        # Check if this is a file path, if not, search for the file path first
        pdf_path = ensure_file_paths([pdf_path], computer)[0]

        return await pdf_summarizer_agent(pdf_path)

    return Tool(
        name=TinyAgentToolName.SUMMARIZE_PDF.value,
        func=summarize_pdf,
        description=(
            f"{TinyAgentToolName.SUMMARIZE_PDF.value}(pdf_path: str) -> str\n"
            " - Summarizes the content of a PDF file and returns the summary.\n"
            " - This tool can only be used AFTER calling open_and_get_file_path tool to get the PDF file path.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.SUMMARIZE_PDF.value}({args[0]})",
    )


async def call_compose_email_agent(
    compose_email_agent: ComposeEmailAgent,
    context: str,
    email_thread: str = "",
    mode: ComposeEmailMode = ComposeEmailMode.NEW,
) -> str:
    return await compose_email_agent(
        context=context, email_thread=email_thread, mode=mode
    )


def get_compose_new_email_tool(
    computer: Computer, compose_email_agent: ComposeEmailAgent
) -> Tool:
    async def compose_new_email(
        recipients: list[str],
        cc: list[str],
        subject: str,
        context: str,
        attachments: list[str],
    ) -> str:
        # Ensure consistent formatting of recipients and cc and ensure they are email addresses
        recipients = ensure_email_formatting(
            recipients if isinstance(recipients, list) else [recipients], computer
        )
        cc = ensure_email_formatting(cc if isinstance(cc, list) else [cc], computer)

        if isinstance(attachments, str):
            attachments = [attachments]

        # Ensure the attachment is a file path, if not search for the file path first
        attachments = ensure_file_paths(attachments, computer)

        body = await call_compose_email_agent(
            compose_email_agent=compose_email_agent, context=context
        )

        return computer.mail.compose_email(
            recipients=recipients,
            cc=cc,
            subject=subject,
            content=body,
            attachments=attachments,
        )

    return Tool(
        name=TinyAgentToolName.COMPOSE_NEW_EMAIL.value,
        func=compose_new_email,
        description=(
            f"{TinyAgentToolName.COMPOSE_NEW_EMAIL.value}("
            "recipients: list[str], "
            "cc: list[str], "
            "subject: str, "
            "context: str, "
            "attachments: list[str]"
            ") -> str\n"
            " - Composes a new email and returns the status of the email composition.\n"
            " - The recipients and cc parameters can be a single email or a list of emails.\n"
            " - The attachments parameter can be a single file path or a list of file paths.\n"
            " - The context parameter is optional and should only be used to pass down some intermediate result. Otherwise, just leave it as empty string.\n"
            " - You MUST put a map location in the 'context' parameter if you want to send a location in the email.\n"
            " - If you want to send a zoom link, you MUST put the zoom link in the 'context' parameter.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.COMPOSE_NEW_EMAIL.value}("
            f"recipients={args[0]}, "
            f"cc={args[1]}, "
            f"subject={args[2]}, "
            f"context={args[3]}, "
            f"attachments={args[4]})"
        ),
    )


def get_reply_to_email_tool(
    computer: Computer, compose_email_agent: ComposeEmailAgent
) -> Tool:
    async def reply_to_email(
        cc: list[str],
        context: str,
        attachments: list[str],
    ) -> str:
        # Ensure consistent formatting of cc and ensure they are email addresses
        cc = ensure_email_formatting(cc if isinstance(cc, list) else [cc], computer)
        if isinstance(attachments, str):
            attachments = [attachments]

        attachments = ensure_file_paths(attachments, computer)

        email_thread = computer.mail.get_email_content()
        context = await call_compose_email_agent(
            compose_email_agent=compose_email_agent,
            context=context,
            email_thread=email_thread,
            mode=ComposeEmailMode.REPLY,
        )

        return computer.mail.reply_to_email(
            cc=cc,
            content=context,
            attachments=attachments,
        )

    return Tool(
        name=TinyAgentToolName.REPLY_TO_EMAIL.value,
        func=reply_to_email,
        description=(
            f"{TinyAgentToolName.REPLY_TO_EMAIL.value}("
            "cc: list[str], "
            "context: str, "
            "attachments: list[str]"
            ") -> str\n"
            " - Replies to the currently selected email in Mail with the given content.\n"
            " - The cc parameter can be a single email or a list of emails.\n"
            " - The attachments parameter can be a single file path or a list of file paths.\n"
            " - The context parameter is optional and should only be used to pass down some intermediate result. Otherwise, just leave it as empty string.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.REPLY_TO_EMAIL.value}("
            f"cc={args[0]}, "
            f"content={args[1]}, "
            f"attachments={args[2]})"
        ),
    )


def get_forward_email_tool(computer: Computer) -> Tool:
    async def forward_email(
        recipients: list[str],
        cc: list[str],
        context: str,
        attachments: list[str],
    ) -> str:
        # Ensure consistent formatting of recipients and cc and ensure they are email addresses
        recipients = ensure_email_formatting(
            recipients if isinstance(recipients, list) else [recipients], computer
        )
        cc = ensure_email_formatting(cc if isinstance(cc, list) else [cc], computer)

        if isinstance(attachments, str):
            attachments = [attachments]

        return computer.mail.forward_email(
            recipients=recipients,
            cc=cc,
            attachments=attachments,
        )

    return Tool(
        name=TinyAgentToolName.FORWARD_EMAIL.value,
        func=forward_email,
        description=(
            f"{TinyAgentToolName.FORWARD_EMAIL.value}("
            "recipients: list[str], "
            "cc: list[str], "
            "context: str, "
            "attachments: list[str]"
            ") -> str\n"
            " - Forwards the currently selected email in Mail with the given content.\n"
            " - The recipients and cc parameters can be a single email or a list of emails.\n"
            " - The attachments parameter can be a single file path or a list of file paths.\n"
            " - The context parameter is optional and should only be used to pass down some intermediate result. Otherwise, just leave it as empty string.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.FORWARD_EMAIL.value}("
            f"recipients={args[0]}, "
            f"cc={args[1]}, "
            f"content={args[2]}, "
            f"attachments={args[3]})"
        ),
    )


def get_maps_open_location_tool(computer: Computer) -> Tool:
    async def maps_open_location(location: str) -> str:
        return computer.maps.open_location(location)

    return Tool(
        name=TinyAgentToolName.MAPS_OPEN_LOCATION.value,
        func=maps_open_location,
        description=(
            f"{TinyAgentToolName.MAPS_OPEN_LOCATION.value}(location: str) -> str\n"
            " - Opens the specified location in Apple Maps.\n"
            " - The query can be a place name, address, or coordinates.\n"
            " - Returns the URL of the location in Apple Maps.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.MAPS_OPEN_LOCATION.value}({args[0]})",
    )


def get_maps_show_directions_tool(computer: Computer) -> Tool:
    async def maps_show_directions(
        start_location: str, end_location: str, transport: str
    ) -> str:
        args: dict[str, Any] = {
            "end": end_location,
        }

        if start_location is not None and len(start_location) > 0:
            args["start"] = start_location
        if (
            transport is not None
            and len(transport) > 0
            and transport in TransportationOptions._value2member_map_
        ):
            args["transport"] = TransportationOptions(transport)
        return computer.maps.show_directions(**args)

    return Tool(
        name=TinyAgentToolName.MAPS_SHOW_DIRECTIONS.value,
        func=maps_show_directions,
        description=(
            f"{TinyAgentToolName.MAPS_SHOW_DIRECTIONS.value}("
            "start_location: str, "
            "end_location: str, "
            "transport: str"
            ") -> str\n"
            " - Show directions from a start location to an end location in Apple Maps.\n"
            " - The transport parameter defaults to 'd' (driving), but can also be 'w' (walking) or 'r' (public transit).\n"
            " - The start location can be left empty to default to the current location of the device.\n"
            " - Returns the URL of the location and directions in Apple Maps.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.MAPS_SHOW_DIRECTIONS.value}("
            f"start_location={args[0]}, "
            f"end_location={args[1]}, "
            f"transport={args[2]})"
        ),
    )


def get_create_note_tool(computer: Computer, notes_agent: NotesAgent) -> Tool:
    async def create_new_note(name: str, content: str, folder: str) -> str:
        return computer.notes.create_note(
            name, await notes_agent(name, content, mode=NotesMode.NEW), folder
        )

    return Tool(
        name=TinyAgentToolName.CREATE_NOTE.value,
        func=create_new_note,
        description=(
            f"{TinyAgentToolName.CREATE_NOTE.value}("
            "name: str, "
            "content: str, "
            "folder: str"
            ") -> str\n"
            " - Creates a new note with the given content.\n"
            " - The name is used as the title of the note.\n"
            " - The content is the main text of the note.\n"
            " - The folder is optional, use an empty string if not applicable.\n"
            " - Returns the status of the note creation.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.CREATE_NOTE.value}("
            f"name={args[0]}, "
            f"content={args[1]}, "
            f"folder={args[2]})"
        ),
    )


def get_open_note_tool(computer: Computer) -> Tool:
    async def open_note(name: str, folder: str) -> str:
        return computer.notes.open_note(name, folder, return_content=True)

    return Tool(
        name=TinyAgentToolName.OPEN_NOTE.value,
        func=open_note,
        description=(
            f"{TinyAgentToolName.OPEN_NOTE.value}(name: str, folder: str) -> str\n"
            " - Opens an existing note by its name.\n"
            " - If a folder is specified, the note is created in that folder; otherwise, it's created in the default folder.\n"
            " - Returns the content of the note.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.OPEN_NOTE.value}(name={args[0]}, folder={args[1]})",
    )


def get_append_note_content_tool(computer: Computer, notes_agent: NotesAgent) -> Tool:
    async def append_note_content(name: str, content: str, folder: str) -> str:
        prev_content = computer.notes.open_note(name, folder, return_content=True)
        return computer.notes.append_to_note(
            name,
            await notes_agent(
                name, content, prev_content=prev_content, mode=NotesMode.APPEND
            ),
            folder,
        )

    return Tool(
        name=TinyAgentToolName.APPEND_NOTE_CONTENT.value,
        func=append_note_content,
        description=(
            f"{TinyAgentToolName.APPEND_NOTE_CONTENT.value}("
            "name: str, "
            "content: str, "
            "folder: str"
            ") -> str\n"
            " - Appends content to an existing note.\n"
            " - If a folder is specified, the note is created in that folder; otherwise, it's created in the default folder.\n"
            " - Returns the status of the content appending.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.APPEND_NOTE_CONTENT.value}("
            f"name={args[0]}, "
            f"content={args[1]}, "
            f"folder={args[2]})"
        ),
    )


def get_create_reminder_tool(computer: Computer) -> Tool:
    async def create_reminder(
        name: str,
        due_date: str,
        notes: str,
        list_name: str,
        priority: int,
        all_day: bool,
    ) -> str:
        # Check that the due date is in the correct format or whether it's provided
        # If not, default to the current time
        due_date_args = get_datetime(due_date)
        if due_date_args is None:
            due_date_args = datetime.datetime.now()

        return computer.reminders.create_reminder(
            name=name,
            due_date=datetime.datetime.fromisoformat(due_date),
            notes=notes,
            list_name=list_name,
            priority=priority,
            all_day=all_day,
        )

    return Tool(
        name=TinyAgentToolName.CREATE_REMINDER.value,
        func=create_reminder,
        description=(
            f"{TinyAgentToolName.CREATE_REMINDER.value}("
            "name: str, "
            "due_date: str, "
            "notes: str, "
            "list_name: str, "
            "priority: int, "
            "all_day: bool"
            ") -> str\n"
            " - Creates a new reminder and returns the status of the reminder creation.\n"
            " - The format for due_date is 'YYYY-MM-DD HH:MM:SS'. "
            "If 'all_day' is True, then the format is 'YYYY-MM-DD'.\n"
            " - The list_name is optional, use an empty string if not applicable.\n"
            " - The priority is optional and defaults to 0.\n"
            " - The all_day parameter is optional and defaults to False.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.CREATE_REMINDER.value}("
            f"name={args[0]}, "
            f"due_date={args[1]}, "
            f"notes={args[2]}, "
            f"list_name={args[3]}, "
            f"priority={args[4]}, "
            f"all_day={args[5]})"
        ),
    )


def get_send_sms_tool(computer: Computer) -> Tool:
    async def send_sms(recipients: list[str], message: str) -> str:
        # Ensure consistent formatting of recipients and ensure they are phone numbers
        recipients = ensure_phone_number_formatting(
            recipients if isinstance(recipients, list) else [recipients], computer
        )
        return computer.sms.send(to=recipients, message=message)

    return Tool(
        name=TinyAgentToolName.SEND_SMS.value,
        func=send_sms,
        description=(
            f"{TinyAgentToolName.SEND_SMS.value}(recipients: list[str], message: str) -> str\n"
            " - Send an SMS to a list of phone numbers.\n"
            " - The recipients argument can be a single phone number or a list of phone numbers.\n"
            " - Returns the status of the SMS.\n"
        ),
        stringify_rule=lambda args: f"{TinyAgentToolName.SEND_SMS.value}(recipients={args[0]}, message={args[1]})",
    )


def get_zoom_meeting_link_tool(
    computer: Computer, zoom_access_token: str | None
) -> Tool:
    if zoom_access_token is None or len(zoom_access_token) <= 0:
        raise ValueError(
            "Couldn't find the Zoom access token. Please provide it in the settings.",
        )

    # Add zoom tool to computer
    computer.zoom = Zoom(zoom_access_token)

    async def get_zoom_meeting_link(
        topic: str,
        start_time: str,
        duration: int,
        meeting_invitees: list[str],
    ) -> str:
        # Ensure consistent formatting of meeting invitees and ensure they are email addresses
        meeting_invitees = ensure_email_formatting(
            (
                meeting_invitees
                if isinstance(meeting_invitees, list)
                else [meeting_invitees]
            ),
            computer,
        )
        return await computer.zoom.get_meeting_link(
            topic=topic,
            start_time=start_time,
            duration=duration,
            meeting_invitees=meeting_invitees,
        )

    return Tool(
        name=TinyAgentToolName.GET_ZOOM_MEETING_LINK.value,
        func=get_zoom_meeting_link,
        description=(
            f"{TinyAgentToolName.GET_ZOOM_MEETING_LINK.value}("
            "topic: str, "
            "start_time: str, "
            "duration: int, "
            "meeting_invitees: Sequence[str]"
            ") -> str\n"
            " - Creates a Zoom meeting and returns the join URL.\n"
            f" - You need to call {TinyAgentToolName.CREATE_CALENDAR_EVENT.value} to attach the zoom link as a note to the calendar event, or "
            f"{TinyAgentToolName.COMPOSE_NEW_EMAIL.value} to send the link to the invitees in the body of the email.\n"
            " - 'topic', 'start_time', 'duration', and 'meeting_invitees' are required.\n"
            " - 'duration' is in minutes.\n"
            " - The format for start_time is 'YYYY-MM-DD HH:MM:SS'.\n"
        ),
        stringify_rule=lambda args: (
            f"{TinyAgentToolName.GET_ZOOM_MEETING_LINK.value}("
            f"topic={args[0]}, "
            f"start_time={args[1]}, "
            f"duration={args[2]}, "
            f"meeting_invitees={args[3]})"
        ),
    )


APPS_TO_TOOL_NAMES = {
    App.CALENDAR: {TinyAgentToolName.CREATE_CALENDAR_EVENT},
    App.CONTACTS: {
        TinyAgentToolName.GET_PHONE_NUMBER,
        TinyAgentToolName.GET_EMAIL_ADDRESS,
    },
    App.FILES: {
        TinyAgentToolName.OPEN_AND_GET_FILE_PATH,
        TinyAgentToolName.SUMMARIZE_PDF,
    },
    App.MAIL: {
        TinyAgentToolName.COMPOSE_NEW_EMAIL,
        TinyAgentToolName.REPLY_TO_EMAIL,
        TinyAgentToolName.FORWARD_EMAIL,
    },
    App.MAPS: {
        TinyAgentToolName.MAPS_OPEN_LOCATION,
        TinyAgentToolName.MAPS_SHOW_DIRECTIONS,
    },
    App.NOTES: {
        TinyAgentToolName.CREATE_NOTE,
        TinyAgentToolName.OPEN_NOTE,
        TinyAgentToolName.APPEND_NOTE_CONTENT,
    },
    App.REMINDERS: {TinyAgentToolName.CREATE_REMINDER},
    App.SMS: {TinyAgentToolName.SEND_SMS},
    App.ZOOM: {TinyAgentToolName.GET_ZOOM_MEETING_LINK},
}


def get_tool_names_from_apps(apps: Collection[App]) -> Collection[TinyAgentToolName]:
    tool_names = set()
    for app in apps:
        tool_names.update(APPS_TO_TOOL_NAMES[app])
    return tool_names


def get_tiny_agent_tools(
    computer: Computer,
    compose_email_agent: ComposeEmailAgent,
    pdf_summarizer_agent: PDFSummarizerAgent,
    notes_agent: NotesAgent,
    tool_names: Collection[TinyAgentToolName],
    zoom_access_token: str | None = None,
) -> list[Tool | StructuredTool]:
    tools: list[Tool | StructuredTool] = []

    # Calendar tools
    if TinyAgentToolName.CREATE_CALENDAR_EVENT in tool_names:
        tools.append(get_create_calendar_event_tool(computer))
    # Contacts tools
    if TinyAgentToolName.GET_PHONE_NUMBER in tool_names:
        tools.append(get_phone_number_tool(computer))
    if TinyAgentToolName.GET_EMAIL_ADDRESS in tool_names:
        tools.append(get_email_address_tool(computer))
    # Files tools
    if TinyAgentToolName.OPEN_AND_GET_FILE_PATH in tool_names:
        tools.append(get_open_and_get_file_path_tool(computer))
    if TinyAgentToolName.SUMMARIZE_PDF in tool_names:
        tools.append(get_summarize_pdf_tool(computer, pdf_summarizer_agent))
    # Mail tools
    if TinyAgentToolName.COMPOSE_NEW_EMAIL in tool_names:
        tools.append(get_compose_new_email_tool(computer, compose_email_agent))
    if TinyAgentToolName.REPLY_TO_EMAIL in tool_names:
        tools.append(get_reply_to_email_tool(computer, compose_email_agent))
    if TinyAgentToolName.FORWARD_EMAIL in tool_names:
        tools.append(get_forward_email_tool(computer))
    # Maps tools
    if TinyAgentToolName.MAPS_OPEN_LOCATION in tool_names:
        tools.append(get_maps_open_location_tool(computer))
    if TinyAgentToolName.MAPS_SHOW_DIRECTIONS in tool_names:
        tools.append(get_maps_show_directions_tool(computer))
    # Notes tools
    if TinyAgentToolName.CREATE_NOTE in tool_names:
        tools.append(get_create_note_tool(computer, notes_agent))
    if TinyAgentToolName.OPEN_NOTE in tool_names:
        tools.append(get_open_note_tool(computer))
    if TinyAgentToolName.APPEND_NOTE_CONTENT in tool_names:
        tools.append(get_append_note_content_tool(computer, notes_agent))
    # Reminders tools
    if TinyAgentToolName.CREATE_REMINDER in tool_names:
        tools.append(get_create_reminder_tool(computer))
    # SMS tools
    if TinyAgentToolName.SEND_SMS in tool_names:
        tools.append(get_send_sms_tool(computer))
    # Zoom tools
    if TinyAgentToolName.GET_ZOOM_MEETING_LINK in tool_names:
        tools.append(get_zoom_meeting_link_tool(computer, zoom_access_token))

    return tools
