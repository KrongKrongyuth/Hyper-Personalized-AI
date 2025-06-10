from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def test_node(
    state:State
    ):
    """_summary_
    Test node use for testing overall workflow.
    Args:
        state (State): Graph state use for tracking chatbot behavior.
    """
    return {"messages": state["messages"]}

def build_graph(
    state:State
    ):
    """_summary_
    Managing graph structure such as adding node, adding edge, etc.
    """
    graph_builder = StateGraph(state)
    
    # Node
    graph_builder.add_node("test_node", test_node)
    
    # Edge
    graph_builder.add_edge(START, "test_node")
    graph_builder.add_edge("test_node", END)
    
    return graph_builder

def compile_graph(
    graph_builder:StateGraph,
    display:bool = True
    ):
    """_summary_
    Compile graph and display graph as ascii.
    Args:
        graph_builder (StateGraph): Graph builder before compile (already added node/edge)
        display (bool, optional): Display graph as ascii. Defaults to True.
    """
    
    graph = graph_builder.compile()
    if display:
        print(graph.get_graph().draw_ascii())
    
    return graph

if __name__ == "__main__":
    # Input query
    input_query = {
        "messages": "Hello"
    }
    
    # Display graph structure
    print("\n******** Graph display **********")
    graph_builder = build_graph(state=State)
    graph = compile_graph(graph_builder=graph_builder)
    
    # Printout result
    print("\n******** Reslut **********")
    print(graph.invoke(input_query))
    print()