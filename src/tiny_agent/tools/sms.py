import platform
import subprocess

from src.tiny_agent.run_apple_script import run_applescript


class SMS:
    def __init__(self):
        self.messages_app = "Messages"

    def send(self, to: list[str], message: str) -> str:
        """
        Opens an SMS draft to the specified recipient using the Messages app,
        without sending it, by simulating keystrokes.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        to_script = []
        for recipient in to:
            recipient = recipient.replace("\n", "")
            to_script.append(
                f"""
                keystroke "{recipient}"
                delay 0.5
                keystroke return
                delay 0.5
            """
            )
        to_script = "".join(to_script)

        escaped_message = message.replace('"', '\\"').replace("'", "’")

        script = f"""
        tell application "System Events"
            tell application "{self.messages_app}"
                activate
            end tell
            tell process "{self.messages_app}"
                set frontmost to true
                delay 0.5
                keystroke "n" using command down
                delay 0.5
                {to_script}
                keystroke tab
                delay 0.5
                keystroke "{escaped_message}"
            end tell
        end tell
        """
        try:
            run_applescript(script)
            return "SMS draft composed"
        except subprocess.CalledProcessError as e:
            return f"An error occurred while composing the SMS: {str(e)}"
