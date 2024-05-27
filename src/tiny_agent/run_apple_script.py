import subprocess


def run_applescript(script: str) -> str:
    """
    Runs the given AppleScript using osascript and returns the result.
    """
    args = ["osascript", "-e", script]
    return subprocess.check_output(args, universal_newlines=True)


def run_applescript_capture(script: str) -> tuple[str, str]:
    """
    Runs the given AppleScript using osascript, captures the output and error, and returns them.
    """
    args = ["osascript", "-e", script]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    stdout, stderr = result.stdout, result.stderr
    return stdout, stderr


def run_command(command) -> tuple[str, str]:
    """
    Executes a shell command and returns the output.
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout, stderr = result.stdout, result.stderr
    return stdout, stderr
