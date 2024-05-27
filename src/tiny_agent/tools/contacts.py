import platform

from src.tiny_agent.run_apple_script import run_applescript_capture


class Contacts:
    def __init__(self):
        pass

    def get_phone_number(self, contact_name: str) -> str:
        """
        Returns the phone number of a contact by name.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        script = f"""
        tell application "Contacts"
            set thePerson to first person whose name is "{contact_name}"
            set theNumber to value of first phone of thePerson
            return theNumber
        end tell
        """
        stout, stderr = run_applescript_capture(script)
        # If the person is not found, we will try to find similar contacts
        if "Can’t get person" in stderr:
            first_name = contact_name.split(" ")[0]
            names = self.get_full_names_from_first_name(first_name)
            if "No contacts found" in names or len(names) == 0:
                return "No contacts found"
            else:
                # Just find the first person
                return self.get_phone_number(names[0])
        else:
            return stout.replace("\n", "")

    def get_email_address(self, contact_name: str) -> str:
        """
        Returns the email address of a contact by name.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        script = f"""
        tell application "Contacts"
            set thePerson to first person whose name is "{contact_name}"
            set theEmail to value of first email of thePerson
            return theEmail
        end tell
        """
        stout, stderr = run_applescript_capture(script)
        # If the person is not found, we will try to find similar contacts
        if "Can’t get person" in stderr:
            first_name = contact_name.split(" ")[0]
            names = self.get_full_names_from_first_name(first_name)
            if "No contacts found" in names or len(names) == 0:
                return "No contacts found"
            else:
                # Just find the first person
                return self.get_email_address(names[0])
        else:
            return stout.replace("\n", "")

    def get_full_names_from_first_name(self, first_name: str) -> list[str] | str:
        """
        Returns a list of full names of contacts that contain the first name provided.
        """
        if platform.system() != "Darwin":
            return "This method is only supported on MacOS"

        script = f"""
        tell application "Contacts"
            set matchingPeople to every person whose first name contains "{first_name}"
            set namesList to {{}}
            repeat with aPerson in matchingPeople
                set end of namesList to name of aPerson
            end repeat
            return namesList
        end tell
        """
        names, _ = run_applescript_capture(script)
        names = names.strip()
        if len(names) > 0:
            # Turn name into a list of strings
            names = list(map(lambda n: n.strip(), names.split(",")))
            return names
        else:
            return "No contacts found."
