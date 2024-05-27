import datetime
import platform
import subprocess

from src.tiny_agent.run_apple_script import run_applescript, run_applescript_capture


class Calendar:
    def __init__(self):
        self.calendar_app = "Calendar"

    def create_event(
        self,
        title: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        location: str = "",
        invitees: list[str] = [],
        notes: str = "",
        calendar: str | None = None,
    ) -> str:
        """
        Creates a new event with the given title, start date, end date, location, and notes.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        applescript_start_date = start_date.strftime("%B %d, %Y %I:%M:%S %p")
        applescript_end_date = end_date.strftime("%B %d, %Y %I:%M:%S %p")

        # Check if the given calendar parameter is valid
        if calendar is not None:
            script = f"""
            tell application "{self.calendar_app}"
                set calendarExists to name of calendars contains "{calendar}"
            end tell
            """
            exists = run_applescript(script)
            if not exists:
                calendar = self._get_first_calendar()
                if calendar is None:
                    return f"Can't find the calendar named {calendar}. Please try again and specify a valid calendar name."

        # If it is not provded, default to the first calendar
        elif calendar is None:
            calendar = self._get_first_calendar()
            if calendar is None:
                return "Can't find a default calendar. Please try again and specify a calendar name."

        invitees_script = []
        for invitee in invitees:
            invitees_script.append(
                f"""
                make new attendee at theEvent with properties {{email:"{invitee}"}}
            """
            )
        invitees_script = "".join(invitees_script)

        script = f"""
        tell application "System Events"
            set calendarIsRunning to (name of processes) contains "{self.calendar_app}"
            if calendarIsRunning then
                tell application "{self.calendar_app}" to activate
            else
                tell application "{self.calendar_app}" to launch
                delay 1
                tell application "{self.calendar_app}" to activate
            end if
        end tell
        tell application "{self.calendar_app}"
            tell calendar "{calendar}"
                set startDate to date "{applescript_start_date}"
                set endDate to date "{applescript_end_date}"
                set theEvent to make new event at end with properties {{summary:"{title}", start date:startDate, end date:endDate, location:"{location}", description:"{notes}"}}
                {invitees_script}
                switch view to day view
                show theEvent
            end tell
            tell application "{self.calendar_app}" to reload calendars
        end tell
        """

        try:
            run_applescript(script)
            return f"""Event created successfully in the "{calendar}" calendar."""
        except subprocess.CalledProcessError as e:
            return str(e)

    def _get_first_calendar(self) -> str | None:
        script = f"""
            tell application "System Events"
                set calendarIsRunning to (name of processes) contains "{self.calendar_app}"
                if calendarIsRunning is false then
                    tell application "{self.calendar_app}" to launch
                    delay 1
                end if
            end tell
            tell application "{self.calendar_app}"
                set firstCalendarName to name of first calendar
            end tell
            return firstCalendarName
            """
        stdout = run_applescript_capture(script)
        if stdout:
            return stdout[0].strip()
        else:
            return None
