import difflib
import platform
import subprocess

from bs4 import BeautifulSoup

from src.tiny_agent.run_apple_script import run_applescript, run_applescript_capture


class Notes:
    _DEFAULT_FOLDER = "Notes"

    def __init__(self):
        self.notes_app = "Notes"

    def create_note(self, name: str, content: str, folder: str | None = None) -> str:
        """
        Creates a new note with the given content and focuses on it. If a folder is specified, the note
        is created in that folder; otherwise, it's created in the default folder.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        folder_line = self._get_folder_line(folder)
        html_content = content.replace('"', '\\"').replace("'", "’")

        script = f"""
        tell application "{self.notes_app}"
            tell account "iCloud"
                {folder_line}
                    set newNote to make new note with properties {{body:"{html_content}"}}
                end tell
            end tell
            activate
            tell application "System Events"
                tell process "Notes"
                    set frontmost to true
                    delay 0.5 -- wait a bit for the note to be created and focus to be set
                end tell
            end tell
            tell application "{self.notes_app}"
                show newNote
            end tell
        end tell
        """

        try:
            run_applescript(script)
            return "Note created and focused successfully."
        except subprocess.CalledProcessError as e:
            return str(e)

    def open_note(
        self,
        name: str,
        folder: str | None = None,
        return_content: bool = False,
    ) -> str:
        """
        Opens an existing note by its name and optionally returns its content.
        If no exact match is found, attempts fuzzy matching to suggest possible notes.
        If return_content is True, returns the content of the note.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        folder_line = self._get_folder_line(folder)

        # Adjust the script to return content if required
        content_line = (
            "return body of theNote"
            if return_content
            else 'return "Note opened successfully."'
        )

        # Attempt to directly open the note with the exact name and optionally return its content
        script_direct_open = f"""
        tell application "{self.notes_app}"
            tell account "iCloud"
                {folder_line}
                    set matchingNotes to notes whose name is "{name}"
                    if length of matchingNotes > 0 then
                        set theNote to item 1 of matchingNotes
                        show theNote
                        {content_line}
                    else
                        return "No exact match found."
                    end if
                end tell
            end tell
        end tell
        """

        try:
            stdout, _ = run_applescript_capture(script_direct_open)
            if (
                "Note opened successfully" in stdout
                or "No exact match found" not in stdout
            ):
                if return_content:
                    return self._convert_note_to_text(stdout.strip())
                return stdout.strip()  # Successfully opened a note with the exact name

            # If no exact match is found, proceed with fuzzy matching
            note_to_open = self._do_fuzzy_matching(name)

            # Focus the note with the closest matching name after fuzzy matching
            script_focus = f"""
            tell application "{self.notes_app}"
                tell account "iCloud"
                    {folder_line}
                        set theNote to first note whose name is "{note_to_open}"
                        show theNote
                        {content_line}
                    end tell
                end tell
                activate
            end tell
            """
            result = run_applescript(script_focus)
            if return_content:
                return self._convert_note_to_text(result.strip())
            return result.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {str(e)}"

    def append_to_note(
        self, name: str, append_content: str, folder: str | None = None
    ) -> str:
        """
        Appends content to an existing note by its name. If the exact name is not found,
        attempts fuzzy matching to find the closest note.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        folder_line = self._get_folder_line(folder)

        # Try to directly find and append to the note with the exact name
        script_find_note = f"""
        tell application "{self.notes_app}"
            tell account "iCloud"
                {folder_line}
                    set matchingNotes to notes whose name is "{name}"
                    if length of matchingNotes > 0 then
                        set theNote to item 1 of matchingNotes
                        return name of theNote
                    else
                        return "No exact match found."
                    end if
                end tell
            end tell
        end tell
        """

        try:
            note_name, _ = run_applescript_capture(
                script_find_note.format(notes_app=self.notes_app, name=name)
            )
            note_name = note_name.strip()

            if "No exact match found" in note_name or not note_name:
                note_name = self._do_fuzzy_matching(name)
                if note_name == "No notes found after fuzzy matching.":
                    return "No notes found after fuzzy matching."

            # If an exact match is found, append content to the note
            html_append_content = append_content.replace('"', '\\"').replace("'", "’")
            script_append = f"""
            tell application "{self.notes_app}"
                tell account "iCloud"
                    {folder_line}
                        set theNote to first note whose name is "{note_name}"
                        set body of theNote to (body of theNote) & "<br>{html_append_content}"
                        show theNote
                    end tell
                end tell
            end tell
            """

            run_applescript(script_append)
            return f"Content appended to note '{name}' successfully."
        except subprocess.CalledProcessError as e:
            return f"Error: {str(e)}"

    def _get_folder_line(self, folder: str | None) -> str:
        if folder is not None and len(folder) > 0 and self._check_folder_exists(folder):
            return f'tell folder "{folder}"'
        return f'tell folder "{Notes._DEFAULT_FOLDER}"'

    def _do_fuzzy_matching(self, name: str) -> str:
        script_search = f"""
            tell application "{self.notes_app}"
                tell account "iCloud"
                    set noteList to name of every note
                end tell
            end tell
        """
        note_names_str, _ = run_applescript_capture(script_search)
        note_names = note_names_str.split(", ")
        closest_matches = difflib.get_close_matches(name, note_names, n=1, cutoff=0.0)
        if not closest_matches:
            return "No notes found after fuzzy matching."

        note_to_open = closest_matches[0]
        return note_to_open

    def _check_folder_exists(self, folder: str) -> bool:
        # Adjusted script to optionally look for a folder
        folder_check_script = f"""
        tell application "{self.notes_app}"
            set folderExists to false
            set folderName to "{folder}"
            if folderName is not "" then
                repeat with eachFolder in folders
                    if name of eachFolder is folderName then
                        set folderExists to true
                        exit repeat
                    end if
                end repeat
            end if
            return folderExists
        end tell
        """

        folder_exists, _ = run_applescript_capture(folder_check_script)
        folder_exists = folder_exists.strip() == "true"

        return folder_exists

    @staticmethod
    def _convert_note_to_text(note_html: str) -> str:
        """
        Converts an HTML note content to plain text.
        """
        soup = BeautifulSoup(note_html, "html.parser")
        return soup.get_text().strip()
