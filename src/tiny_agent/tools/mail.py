import platform
import subprocess

from src.tiny_agent.run_apple_script import run_applescript, run_applescript_capture


class Mail:
    def __init__(self) -> None:
        self.mail_app: str = "Mail"

    def compose_email(
        self,
        recipients: list[str],
        subject: str,
        content: str,
        attachments: list[str],
        cc: list[str],
    ) -> str:
        """
        Composes a new email with the given recipients, subject, content, and attaches files from the given paths.
        Adds cc recipients if provided. Does not send it but opens the composed email to the user.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        # Format recipients and cc recipients for AppleScript list
        recipients_list = Mail._format_email_addresses(recipients)
        cc_list = Mail._format_email_addresses(cc)
        attachments_str = Mail._format_attachments(attachments)

        content = content.replace('"', '\\"').replace("'", "’")
        script = f"""
        tell application "{self.mail_app}"
            set newMessage to make new outgoing message with properties {{subject:"{subject}", content:"{content}" & return & return}}
            tell newMessage
                repeat with address in {recipients_list}
                    make new to recipient at end of to recipients with properties {{address:address}}
                end repeat
                repeat with address in {cc_list}
                    make new cc recipient at end of cc recipients with properties {{address:address}}
                end repeat
                {attachments_str}
            end tell
            activate
        end tell
        """

        try:
            run_applescript(script)
            return "New email composed successfully with attachments and cc."
        except subprocess.CalledProcessError as e:
            return str(e)

    def reply_to_email(
        self, content: str, cc: list[str], attachments: list[str]
    ) -> str:
        """
        Replies to the currently selected email in Mail with the given content.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        cc_list = Mail._format_email_addresses(cc)
        attachments_str = Mail._format_attachments(attachments)

        content = content.replace('"', '\\"').replace("'", "’")
        script = f"""
        tell application "{self.mail_app}"
            activate
            set selectedMessages to selected messages of message viewer 1
            if (count of selectedMessages) < 1 then
                return "No message selected."
            else
                set theMessage to item 1 of selectedMessages
                set theReply to reply theMessage opening window yes
                tell theReply
                    repeat with address in {cc_list}
                        make new cc recipient at end of cc recipients with properties {{address:address}}
                    end repeat
                    set content to "{content}"
                    {attachments_str}
                end tell
            end if
        end tell
        """

        try:
            run_applescript(script)
            return "Replied to the selected email successfully."
        except subprocess.CalledProcessError as e:
            return "An email has to be viewed in Mail to reply to it."

    def forward_email(
        self, recipients: list[str], cc: list[str], attachments: list[str]
    ) -> str:
        """
        Forwards the currently selected email in Mail to the given recipients with the given content.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        # Format recipients and cc recipients for AppleScript list
        recipients_list = Mail._format_email_addresses(recipients)
        cc_list = Mail._format_email_addresses(cc)
        attachments_str = Mail._format_attachments(attachments)

        script = f"""
        tell application "{self.mail_app}"
            activate
            set selectedMessages to selected messages of message viewer 1
            if (count of selectedMessages) < 1 then
                return "No message selected."
            else
                set theMessage to item 1 of selectedMessages
                set theForward to forward theMessage opening window yes
                tell theForward
                    repeat with address in {recipients_list}
                        make new to recipient at end of to recipients with properties {{address:address}}
                    end repeat
                    repeat with address in {cc_list}
                        make new cc recipient at end of cc recipients with properties {{address:address}}
                    end repeat
                    {attachments_str}
                end tell
            end if
        end tell
        """

        try:
            run_applescript(script)
            return "Forwarded the selected email successfully."
        except subprocess.CalledProcessError as e:
            return "An email has to be viewed in Mail to forward it."

    def get_email_content(self) -> str:
        """
        Gets the content of the currently viewed email in Mail.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        script = f"""
        tell application "{self.mail_app}"
            activate
            set selectedMessages to selected messages of message viewer 1
            if (count of selectedMessages) < 1 then
                return "No message selected."
            else
                set theMessage to item 1 of selectedMessages
                -- Get the content of the message
                set theContent to content of theMessage
                return theContent
            end if
        end tell
        """

        try:
            return run_applescript(script)
        except subprocess.CalledProcessError as e:
            return "No message selected or found."

    def find_and_select_first_email_from(self, sender: str) -> str:
        """
        Finds and selects an email in Mail based on the sender's name.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        script = f"""
        tell application "{self.mail_app}"
            set theSender to "{sender}"
            set theMessage to first message of inbox whose sender contains theSender
            set selected messages of message viewer 1 to {{theMessage}}
            activate
            open theMessage
        end tell
        """

        try:
            run_applescript(script)
            return "Found and selected the email successfully."
        except subprocess.CalledProcessError as e:
            return "No message found from the sender."

    @staticmethod
    def _format_email_addresses(emails: list[str]) -> str:
        return "{" + ", ".join([f'"{email}"' for email in emails]) + "}"

    @staticmethod
    def _format_attachments(attachments: list[str]) -> str:
        attachments_str = []
        for attachment in attachments:
            attachment = attachment.replace('"', '\\"')
            attachments_str.append(
                f"""
                make new attachment with properties {{file name:"{attachment}"}} at after the last paragraph
            """
            )
        return "".join(attachments_str)
