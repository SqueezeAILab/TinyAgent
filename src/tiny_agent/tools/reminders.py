import datetime
import platform
import subprocess

from src.tiny_agent.run_apple_script import run_applescript


class Reminders:
    def __init__(self):
        self.reminders_app = "Reminders"

    def create_reminder(
        self,
        name: str,
        due_date: datetime.datetime | None = None,
        notes: str = "",
        list_name: str = "",
        priority: int = 0,
        all_day: bool = False,
    ) -> str:
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        if due_date is not None:
            due_date_script = f''', due date:date "{due_date.strftime("%B %d, %Y")
                if all_day
                else due_date.strftime("%B %d, %Y %I:%M:%S %p")}"'''
        else:
            due_date_script = ""

        notes = notes.replace('"', '\\"').replace("'", "’")
        script = f"""
        tell application "{self.reminders_app}"
            set listExists to false
            set listName to "{list_name}"
            if listName is not "" then
                repeat with eachList in lists
                    if name of eachList is listName then
                        set listExists to true
                        exit repeat
                    end if
                end repeat
            end if
            if listExists then
                tell list "{list_name}"
                    set newReminder to make new reminder with properties {{name:"{name}", body:"{notes}", priority:{priority}{due_date_script}}}
                    activate
                    show newReminder
                end tell
            else
                set newReminder to make new reminder with properties {{name:"{name}", body:"{notes}", priority:{priority}{due_date_script}}}
                activate
                show newReminder
            end if
        end tell
        """

        try:
            run_applescript(script)
            return f"Reminder '{name}' created successfully in the '{list_name}' list."
        except subprocess.CalledProcessError as e:
            return f"Failed to create reminder: {str(e)}"
