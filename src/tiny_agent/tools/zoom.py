import datetime
from typing import Sequence, TypedDict
from zoneinfo import ZoneInfo

import aiohttp
import dateutil.parser


class Zoom:
    _TIMEZONE = "US/Pacific"
    _MEETINGS_ENDPOINT = "https://api.zoom.us/v2/users/me/meetings"

    class ZoomMeetingInfo(TypedDict):
        """
        Zoom meeting information returned by the create meeting endpoint.
        https://developers.zoom.us/docs/api/rest/reference/zoom-api/methods/#operation/meetingCreate
        Currently only types the fields we need, so add more if needed.
        """

        join_url: str

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token

    async def get_meeting_link(
        self,
        topic: str,
        start_time: str,
        duration: int,
        meeting_invitees: Sequence[str],
    ) -> str:
        """
        Create a new Zoom meeting and return the join URL.
        """
        start_time_utc = (
            dateutil.parser.parse(start_time)
            .replace(tzinfo=ZoneInfo(Zoom._TIMEZONE))
            .astimezone(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        topic = topic[:200]

        resp = await aiohttp.ClientSession().post(
            Zoom._MEETINGS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            },
            json={
                "topic": topic,
                "start_time": start_time_utc,
                "duration": duration,
                "meeting_invitees": meeting_invitees,
            },
        )

        info: Zoom.ZoomMeetingInfo = await resp.json()

        return info["join_url"]
