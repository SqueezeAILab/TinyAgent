import ast
import re
from typing import Any

from src.llm_compiler.constants import JOINNER_FINISH, JOINNER_REPLAN
from src.llm_compiler.task_fetching_unit import Task
from src.utils.data_utils import JoinStep, PlanStep, PlanStepToolName

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\s*\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"


def _parse_llm_compiler_action_args(args: str) -> Any:
    """Parse arguments from a string."""
    # This will convert the string into a python object
    # e.g. '"Ronaldo number of kids"' -> ("Ronaldo number of kids", )
    # '"I can answer the question now.", [3]' -> ("I can answer the question now.", [3])
    if args == "":
        return ()
    try:
        args = ast.literal_eval(args)
    except:
        args = args
    if not isinstance(args, list) and not isinstance(args, tuple):
        args = (args,)  # type: ignore
    return args


def parse_plan(plan: str) -> dict[int, Any]:
    # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
    # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
    pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
    matches = re.findall(pattern, plan)

    graph_dict = {}

    for match in matches:
        # idx = 1, function = "search", args = "Ronaldo number of kids"
        # thought will be the preceding thought, if any, otherwise an empty string
        _, idx, tool_name, args, _ = match
        idx = int(idx)

        # Create a dummy task
        task = Task(
            idx=idx,
            name=tool_name,
            tool=lambda x: None,
            args=_parse_llm_compiler_action_args(args),
            dependencies=[],
            stringify_rule=None,
            thought=None,
            is_join=tool_name == "join",
        )

        graph_dict[idx] = task
        if task.is_join:
            break

    return graph_dict


def get_parsed_planner_output(tasks: dict[int, Any]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    try:
        for _, task in tasks.items():
            step = PlanStep(
                tool_name=(PlanStepToolName(task.name)),
                tool_args=task.args,
            )
            steps.append(step)
    except Exception as e:
        print("Tool hallucination error: ", e)

    return steps


def get_parsed_planner_output_from_raw(raw_answer: str) -> list[PlanStep]:
    tasks = parse_plan(raw_answer)
    return get_parsed_planner_output(tasks)


def get_parsed_joinner_output(raw_answer: str) -> JoinStep:
    thought, answer, is_replan = "", "", False  # default values
    raw_answers = raw_answer.split("\n")
    for ans in raw_answers:
        start_of_answer = ans.find("Action:")
        if start_of_answer >= 0:
            ans = ans[start_of_answer:]
        if ans.startswith("Action:"):
            answer = ans[ans.find("(") + 1 : ans.rfind(")")]
            is_replan = JOINNER_REPLAN in ans
        elif ans.startswith("Thought:"):
            thought = ans.split("Thought:")[1].strip()

    step = JoinStep(
        thought=thought,
        action=JOINNER_REPLAN if is_replan else JOINNER_FINISH,
        message=answer,
    )
    return step


def evaluate_plan(label_plan: list[PlanStep], predicted_plan: list[PlanStep]) -> float:
    """Returns the accuracy of the predicted plan based on the tool names and the right ordering."""
    # Assuming the lengths of the two plans are the same
    correct = 0
    for label_step, predicted_step in zip(label_plan, predicted_plan):
        if label_step.tool_name == predicted_step.tool_name:
            correct += 1
    return correct / len(label_plan)


def evaluate_tool_recall(
    label_plan: list[PlanStep], predicted_plan: list[PlanStep]
) -> float:
    """
    Instead of determining the score based on the correct function call in the plan,
    determines the score based on the correct set of function calls in the plan.
    """
    label_set = set([step.tool_name for step in label_plan])
    predicted_set = set([step.tool_name for step in predicted_plan])

    # Check how many of the predicted steps are in the label set
    correct = len(label_set.intersection(predicted_set))
    return correct / len(label_set)


def evaluate_join(label_join: JoinStep, predicted_join: JoinStep) -> float:
    """Returns 1.0 if the predicted join step is correct, else 0.0."""
    if label_join.action == predicted_join.action:
        return 1.0
    return 0.0
