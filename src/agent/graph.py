"""
Vanilla Python state machine assembly for the U-Bahn operator agent.
Replaces LangGraph to remove third-party dependencies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import AgentState
from agent.nodes import (
    intent_extractor_node,
    fingerprinter_node,
    tool_planner_node,
    tool_executor_node,
    synthesizer_node,
    response_formatter_node,
)

class SimpleAgentGraph:
    """
    A simple linear execution graph that runs nodes in sequence.
    Replaces LangGraph's StateGraph.
    """
    def __init__(self):
        self.nodes = [
            intent_extractor_node,
            fingerprinter_node,
            tool_planner_node,
            tool_executor_node,
            synthesizer_node,
            response_formatter_node
        ]

    def invoke(self, initial_state: dict) -> dict:
        state = dict(initial_state)
        for node_func in self.nodes:
            update = node_func(state)
            # Merge update into state
            for k, v in update.items():
                # For messages, we'd normally append. Here we just overwrite
                # or handle list extension if it's the messages key.
                if k == "messages" and "messages" in state:
                    state["messages"].extend(v)
                else:
                    state[k] = v
        return state

def build_agent_graph():
    """
    Build and return the Vanilla Python agent graph.
    """
    return SimpleAgentGraph()
