"""
Base Agent Class

Implement Duplicate functions of Agents.
- Initialize/Add Nodes in graph
- INitialize/Add Edges in graph
- Set State/Input/Ouput Schema
- Compile StateGraph Instance
"""

from abc import ABC, abstractmethod
from typing import Dict, ClassVar


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph



class BaseAgent(ABC):
    """
    Base Agent Class
    """
    NODE_NAMES: ClassVar[Dict[str, str]] = {}

    def __init__(
        self,
        model: BaseChatModel | ChatOpenAI,
        state_schema: type,
        input_schema: type,
        output_schema: type,
        agent_name: str
    ) -> None:
        """
        Initialize Base LangGraph Agent.

        Args:
            model (BaseChatModel | ChatOpenAI) : LLM Instance that will be use in agent
            state_schema (type) : graph state class. (TypedDict or BaseModel)
            input_schema (type) : graph state class. (TypedDict or BaseModel)
            output_schema (type) : graph state class. (TypedDict or BaseModel)
            agent_name (str) : name of agent.
        """
        self.model = model
        self.state_schema = state_schema
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.agent_name = agent_name
    
    # ------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------
    def get_node_name(self, key: str = "DEFAULT") -> str:
        """
        Get name of certain node that set in langgraph.

        Args:
            key (str) : 'key' of certain node.
        """

        if key not in self.NODE_NAMES:
            raise ValueError(f"Node Name {key} not found in {self.NODE_NAMES}")
        
        return self.NODE_NAMES[key]
    
    # ------------------------------------------------------------
    # Agent Core Process (Should Implement in Subclass)
    # ------------------------------------------------------------
    @abstractmethod
    def _init_nodes(
        self,
        graph:StateGraph
    ) -> None:
        """
        Initialize/Add Nodes in graph.

        Args:
            graph (StateGraph) : Graph instance of langgraph StateGraph Class
        """

        raise NotImplementedError(f"'_init_nodes' method must be implemented before using.")
    
    @abstractmethod
    def _init_edges(
        self,
        graph:StateGraph
    ) -> None:
        """
        Initialize/Add Edges in graph.

        Args:
            graph (StateGraph) : Graph instance of langgraph StateGraph class
        """

        raise NotImplementedError(f"'_init_edges' method must be implemented before using.")
    
    # ------------------------------------------------------------
    # Agent Core Process (Should Implement in Subclass)
    # ------------------------------------------------------------
    def build_graph(self):
        """
        Compile StateGraph instance.
        """

        _graph = StateGraph(
            state_schema=self.state_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema
        )

        self._init_nodes(_graph)
        self._init_edges(_graph)

        self.graph = _graph.compile(name=self.agent_name)