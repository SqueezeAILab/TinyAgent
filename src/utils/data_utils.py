import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from src.llm_compiler.constants import JOINNER_FINISH, JOINNER_REPLAN


class PlanStepToolName(Enum):
    GET_PHONE_NUMBER = "get_phone_number"
    GET_EMAIL_ADDRESS = "get_email_address"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    OPEN_AND_GET_FILE_PATH = "open_and_get_file_path"
    SUMMARIZE_PDF = "summarize_pdf"
    COMPOSE_NEW_EMAIL = "compose_new_email"
    REPLY_TO_EMAIL = "reply_to_email"
    FORWARD_EMAIL = "forward_email"
    MAPS_OPEN_LOCATION = "maps_open_location"
    MAPS_SHOW_DIRECTIONS = "maps_show_directions"
    CREATE_NOTE = "create_note"
    OPEN_NOTE = "open_note"
    APPEND_NOTE_CONTENT = "append_note_content"
    CREATE_REMINDER = "create_reminder"
    SEND_SMS = "send_sms"
    GET_ZOOM_MEETING_LINK = "get_zoom_meeting_link"
    JOIN = "join"


class DataPointType(Enum):
    PLAN = "plan"
    JOIN = "join"
    REPLAN = "replan"


@dataclass
class PlanStep:
    tool_name: PlanStepToolName
    tool_args: dict[str, Any]

    def serialize(self) -> dict[str, Any]:
        return {
            "tool_name": (
                self.tool_name.value
                if isinstance(self.tool_name, PlanStepToolName)
                else "join"
            ),
            "tool_args": self.tool_args,
        }


@dataclass
class JoinStep:
    thought: str
    action: Literal[JOINNER_FINISH, JOINNER_REPLAN]  # type: ignore
    # Message is only applicable if the action is "Finish"
    message: str

    def serialize(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "message": self.message,
        }


@dataclass
class DataPoint:
    type: DataPointType
    raw_input: str
    raw_output: str
    parsed_output: list[PlanStep] | JoinStep
    id: uuid.UUID
    index: int

    def serialize(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "parsed_output": (
                [step.serialize() for step in self.parsed_output]
                if isinstance(self.parsed_output, list)
                else self.parsed_output.serialize()
            ),
            "id": str(self.id),
            "index": self.index,
        }


@dataclass
class Data:
    input: str
    output: list[DataPoint]
    closest_5_queries: list[str] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": [point.serialize() for point in self.output],
            "closest_5_queries": self.closest_5_queries,
        }


def deserialize_data(data_json: dict[str, Any]) -> dict[str, Data]:
    data_objects = {}

    for key, value in data_json.items():
        # Assuming 'input' and 'output' are directly accessible within `value`
        input_str = value["input"]
        output_list = value["output"]

        data_points = []
        for output in output_list:
            parsed_output = output["parsed_output"]
            data_point = DataPoint(
                type=DataPointType(output["type"]),
                raw_input=output["raw_input"],
                raw_output=output["raw_output"],
                parsed_output=(
                    (
                        [
                            PlanStep(
                                tool_name=(PlanStepToolName(step["tool_name"])),
                                tool_args=step["tool_args"],
                            )
                            for step in parsed_output
                        ]
                        if isinstance(parsed_output, list)
                        else JoinStep(
                            thought=parsed_output["thought"],
                            action=parsed_output["action"],
                            message=parsed_output["message"],
                        )
                    )
                ),
                id=uuid.UUID(output["id"]),
                index=output["index"],
            )

            data_points.append(data_point)

        data_objects[key] = Data(
            input=input_str,
            output=data_points,
        )

        if "closest_5_queries" in value:
            data_objects[key].closest_5_queries = value["closest_5_queries"]

    return data_objects


def save_data(data_objects: dict[str, Any], json_path: str) -> None:
    data_json = {}

    for key, data in data_objects.items():
        data_json[key] = data.serialize()

    with open(json_path, "w") as file:
        json.dump(data_json, file, indent=4)


def initialize_data_objects(json_path: str) -> dict[str, Data]:
    if not os.path.exists(json_path):
        with open(json_path, "w") as file:
            file.write("{}")
        data_objects = {}
    else:
        with open(json_path, "r") as file:
            data_objects = json.load(file)

        data_objects = deserialize_data(data_objects)
    return data_objects
