import re
from typing import Any

import networkx as nx

from src.utils.data_utils import PlanStep

DEPENDENCY_REGEX = r"\$\d+"


def check_for_dependency(item: str) -> str | None:
    match = re.search(DEPENDENCY_REGEX, item)
    if match:
        return match.group()
    return None


def build_graph(plan: list[PlanStep]) -> nx.DiGraph:
    function_calls = [step.serialize() for step in plan]
    graph = nx.DiGraph()

    def add_new_dependency_edge(
        *,
        node_name: str,
        dependency: str,
    ) -> tuple[dict[str, Any], str]:
        dependency_index = int(dependency[1:]) - 1
        dependency_func = function_calls[dependency_index]
        dependency_node_name = f"{dependency_index+1}: {dependency_func['tool_name']}"
        graph.add_edge(dependency_node_name, node_name)

        return dependency_func, dependency_node_name

    def add_dependencies(dependencies: list[Any], node_name: str):
        nested_deps = []
        for dep in dependencies:
            if isinstance(dep, list):
                # Checking for dependencies in list-type arguments
                for item in dep:
                    if (
                        isinstance(item, str)
                        and (item := check_for_dependency(item)) is not None
                    ):
                        dependency_func, dependency_node_name = add_new_dependency_edge(
                            node_name=node_name, dependency=item
                        )

                        # In order to avoid infinite recursion, we check
                        # if the dependency node is the same as the current node
                        if dependency_node_name == node_name:
                            raise ValueError(
                                "Circular dependency detected. "
                                "The tool is dependent on itself, which is not allowed."
                            )

                        # Recursively add nested dependencies
                        dep_dict = {
                            "tool_name": dependency_func["tool_name"],
                            "dependencies": (
                                add_dependencies(
                                    dependency_func["tool_args"], dependency_node_name
                                )
                                if dependency_node_name != node_name
                                else []
                            ),
                        }
                        nested_deps.append(dep_dict)
            elif (
                isinstance(dep, str) and (dep := check_for_dependency(dep)) is not None
            ):
                dependency_func, dependency_node_name = add_new_dependency_edge(
                    node_name=node_name, dependency=dep
                )

                # In order to avoid infinite recursion, we check
                # if the dependency node is the same as the current node
                if dependency_node_name == node_name:
                    raise ValueError(
                        "Circular dependency detected. "
                        "The tool is dependent on itself, which is not allowed."
                    )

                # Recursively add nested dependencies
                # In order to avoid infinite recursion, we check if the dependency node is the same as the current node
                dep_dict = {
                    "tool_name": dependency_func["tool_name"],
                    "dependencies": (
                        add_dependencies(
                            dependency_func["tool_args"], dependency_node_name
                        )
                        if dependency_node_name != node_name
                        else []
                    ),
                }
                nested_deps.append(dep_dict)
        return nested_deps

    for index, func in enumerate(function_calls):
        node_name = f"{index+1}: {func['tool_name']}"
        graph.add_node(
            node_name, tool_name=func["tool_name"], tool_args=func["tool_args"]
        )
        dependencies = add_dependencies(func["tool_args"], node_name)
        graph.nodes[node_name]["dependencies"] = dependencies

    return graph


def node_match(n1: dict, n2: dict) -> bool:
    if n1["tool_name"] != n2["tool_name"]:
        return False

    # Compare dependency structures recursively
    def compare_dependencies(d1, d2):
        if len(d1) != len(d2):
            return False
        for item1, item2 in zip(
            sorted(d1, key=lambda x: x["tool_name"]),
            sorted(d2, key=lambda x: x["tool_name"]),
        ):
            if not (
                item1["tool_name"] == item2["tool_name"]
                and compare_dependencies(item1["dependencies"], item2["dependencies"])
            ):
                return False
        return True

    return compare_dependencies(n1["dependencies"], n2["dependencies"])


def compare_graphs_with_success_rate(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """Ouputs 1.0 if the graphs are isomorphic, 0.0 otherwise."""
    if nx.is_isomorphic(graph1, graph2, node_match=node_match):
        return 1.0
    return 0.0


def compare_graphs_with_edit_distance(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """
    Calculates the graph edit distance between two graphs.
    """
    f_node_match = lambda n1, n2: node_match(n1, n2)

    # First, check for isomoprhism since it is a faster check. If yes, the GED is 0.
    if nx.is_isomorphic(graph1, graph2, node_match=f_node_match):
        return 0.0

    # Calculate the graph edit distance
    ged = nx.graph_edit_distance(
        graph1,
        graph2,
        node_match=f_node_match,
        timeout=10,
    )

    return ged
