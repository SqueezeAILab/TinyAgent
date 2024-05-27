import difflib
import os  # Import os to check if the path exists
import platform
import subprocess

from src.tiny_agent.run_apple_script import run_command


class SpotlightSearch:
    def __init__(self):
        pass

    def open(self, name_or_path: str) -> str:
        """
        Does Spotlight Search and opens the first thing that matches the name.
        If no exact match, performs fuzzy search.
        Additionally, if the input is a path, tries to open the file directly.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        # Check if input is a path and file exists
        if name_or_path.startswith("/") and os.path.exists(name_or_path):
            try:
                subprocess.run(["open", name_or_path])
                return name_or_path
            except Exception as e:
                return f"Error opening file: {e}"

        # Use mdfind for fast searching with Spotlight
        command_search_exact = ["mdfind", f"kMDItemDisplayName == '{name_or_path}'"]
        stdout, _ = run_command(command_search_exact)

        if stdout:
            paths = stdout.strip().split("\n")
            path = paths[0] if paths else None
            if path:
                subprocess.run(["open", path])
                return path

        # If no exact match, perform fuzzy search on the file names
        command_search_general = ["mdfind", name_or_path]
        stdout, stderr = run_command(command_search_general)

        paths = stdout.strip().split("\n") if stdout else []

        if paths:
            best_match = difflib.get_close_matches(name_or_path, paths, n=1, cutoff=0.0)
            if best_match:
                _, stderr = run_command(["open", best_match[0]])
                if len(stderr) > 0:
                    return f"Error: {stderr}"
                return best_match[0]
            else:
                return "No file found after fuzzy matching."
        else:
            return "No file found with exact or fuzzy name."
