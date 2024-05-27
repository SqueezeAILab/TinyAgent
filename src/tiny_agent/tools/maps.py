import webbrowser
from urllib.parse import quote_plus

from src.tiny_agent.models import TransportationOptions


class Maps:
    def __init__(self):
        pass

    def open_location(self, query: str):
        """
        Opens the specified location in Apple Maps.
        The query can be a place name, address, or coordinates.
        """
        base_url = "https://maps.apple.com/?q="
        query_encoded = quote_plus(query)
        full_url = base_url + query_encoded
        webbrowser.open(full_url)
        return f"Location of {query} in Apple Maps: {full_url}"

    def show_directions(
        self,
        end: str,
        start: str = "",
        transport: TransportationOptions = TransportationOptions.DRIVING,
    ):
        """
        Shows directions from a start location to an end location in Apple Maps.
        The transport parameter defaults to 'd' (driving), but can also be 'w' (walking) or 'r' (public transit).
        The start location can be left empty to default to the current location of the device.
        """
        base_url = "https://maps.apple.com/?"
        if len(start) > 0:
            start_encoded = quote_plus(start)
            start_param = f"saddr={start_encoded}&"
        else:
            start_param = ""  # Use the current location
        end_encoded = quote_plus(end)
        transport_flag = f"dirflg={transport.value}"
        full_url = f"{base_url}{start_param}daddr={end_encoded}&{transport_flag}"
        webbrowser.open(full_url)
        return f"Directions to {end} in Apple Maps: {full_url}"
