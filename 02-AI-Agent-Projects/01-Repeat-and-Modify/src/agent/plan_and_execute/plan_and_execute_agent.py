"""
Agent Run with 'Plan and Execute' Architecture.
"""

from src.agent.base.base_agent import BaseAgent
from src.prompt.prompt_manager import PromptManager

import os
from operator import add
from typing import List, Union, Annotated, Tuple, Dict, ClassVar
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# ------------------------------------------------------------------------------------------------
# Response Schema of Language Model
# ------------------------------------------------------------------------------------------------
class PlanSchema(BaseModel):
    """
    사용자 요청에 답변하기 위해 수립된 계획을 반환하기 위해 이 답변구조를 사용하세요.
    """

    steps: List[str] = Field(
        description="사용자의 요청을 완수하기 위한 일련의 계획 리스트. 반드시 실행 순서대로 작성되어야 함."
    )

class ResponseSchema(BaseModel):
    """
    사용자 요청에 답변하기 위해서는 이 답변 형태를 사용하세요.
    """

    answer: str = Field(
        description="사용자 요청에 따른 답변"
    )

class ReplannerDecisionSchema(BaseModel):
    replanner_decision: Union[PlanSchema, ResponseSchema] = Field(
        description="""
다음으로 취해질 행동입니다.
현재 생성된 답변으로 사용자에게 답변하기 위해서는 ResponseSchema 를 사용하세요.
이 후 단계를 더 수행하거나, 도구를 사용하여 답변 개선이 필요하다면 PlanSchema 를 사용하세요.
"""
    )

# ------------------------------------------------------------------------------------------------
# Graph Schema
# ------------------------------------------------------------------------------------------------
class InputSchema(TypedDict):
    request: str

class OutputSchema(TypedDict):
    answer: str

class StateSchema(InputSchema, OutputSchema):
    plan: List[str]
    trajectory: Annotated[List[Tuple], add]

# ------------------------------------------------------------------------------------------------
# Plan And Execute Agent Class
# ------------------------------------------------------------------------------------------------
class PlanAndExecuteAgent(BaseAgent):
    """
    PlanAndExecute Agent Class.
    """

    NODE_NAMES: ClassVar[Dict[str, str]] = {
        "ACTOR" : "actor",
        "PLANNER" : "planner",
        "REPLANNER" : "replanner"
    }

    def __init__(
        self,
        model: BaseChatModel | ChatOpenAI,
        state_schema: type,
        input_schema: type,
        output_shcema: type,
        agent_name: str,
    ) -> None:
        """
        Initialize Agent with 'plan and execute' architecture.

        Args:
            model: BaseChatModel | ChatOpenAI
            state_schema: type
            input_schema: type
            output_shcema: type
            agent_name: str
            actor_prompt: None | str = None
            planner_prompt: None | str = None
            replanner_prompt: None | str = None
        """

        super().__init__(
            model=model,
            state_schema=state_schema,
            input_schema=input_schema,
            output_schema=output_shcema,
            agent_name=agent_name
        )

        self.prompt_manager = PromptManager(prompts_dir=os.path.join('src', 'agent', 'plan_and_execute', 'prompts'))
        
    # ------------------------------------------------------------------------------------------------
    # Create Agent Instances
    # ------------------------------------------------------------------------------------------------
    def _get_actor_agent(self):
        actor_system_prompt = self.prompt_manager.load_prompt('actor')

        search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced"
        )
        tools = [search_tool]

        actor_agent = create_react_agent(
            model=self.model,
            tools=tools,
            prompt=actor_system_prompt,
        )

        return actor_agent
    
    def _get_planner_agent(self):
        planner_prompt = self.prompt_manager.load_prompt('planner')
        planner_agent = planner_prompt | self.model.with_structured_output(PlanSchema)

        return planner_agent

    def _get_replanner_agent(self):
        replanner_prompt= self.prompt_manager.load_prompt('replanner')

        replanner_agent = replanner_prompt | self.model.with_structured_output(ReplannerDecisionSchema)

        return replanner_agent
    
    
    def _get_actor_node(self):
        actor_agent = self._get_actor_agent()

        def actor(state:StateSchema):
            steps = state["plan"]
            user_request = state["request"]

            task_list = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
            current_step = f"\n\n이 중에서 다음 과업을 수행해주세요.: '{steps[0]}'"
            prompt = "당신이 수행할 행동 계획은: \n" + task_list + current_step

            response = actor_agent.invoke({"messages" : [HumanMessage(content=prompt)]})

            return {
                "request" : user_request,
                "plan" : steps,
                "trajectory" : [(steps[0], response["messages"][-1].content)]
            }

        return actor

    def _get_planner_node(self):
        planner_agent = self._get_planner_agent()

        def planner(state:InputSchema):
            user_request = state.get('request')

            response = planner_agent.invoke({"planning_request" : [HumanMessage(content=user_request)]})

            return {
                "request" : user_request,
                "plan" : response.steps
            }
        
        return planner
    
    def _get_replanner_node(self):
        replanner_agent = self._get_replanner_agent()

        def replanner(state:StateSchema):
            trajectory = ""
            for idx, content in enumerate(state.get('trajectory')):
                step, result = content[0], content[1]
                trajectory += f"Step {idx+1} : {step}\nResult: {result}\n\n"

            response = replanner_agent.invoke({
                "request" : state.get('request'),
                "plan" : state.get('plan'),
                "trajectory" : trajectory
            })

            if isinstance(response.replanner_decision, ResponseSchema):
                return {
                    "answer" : response.replanner_decision.answer
                }
            else:
                state['plan'] = response.replanner_decision.steps
                return state
        
        return replanner
    
    # ------------------------------------------------------------------------------------------------
    # Create Conditional Edge
    # ------------------------------------------------------------------------------------------------
    def _get_should_continue(self):
        def shold_continue(state:StateSchema):
            if "answer" in state and state.get("answer") : return "end"
            else: return "actor"
        
        return shold_continue
    
    # ------------------------------------------------------------------------------------------------
    # Initialize/Add Node in Graph
    # ------------------------------------------------------------------------------------------------
    def _init_nodes(
        self,
        graph:StateGraph
    ) -> None:
        graph.add_node(self.get_node_name("ACTOR"), self._get_actor_node())
        graph.add_node(self.get_node_name("PLANNER"), self._get_planner_node())
        graph.add_node(self.get_node_name("REPLANNER"), self._get_replanner_node())
    
    # ------------------------------------------------------------------------------------------------
    # Initialize/Add Edges in Graph
    # ------------------------------------------------------------------------------------------------
    def _init_edges(
        self,
        graph:StateGraph
    ) -> None:
        graph.add_edge(START, self.get_node_name("PLANNER"))
        graph.add_edge(self.get_node_name("PLANNER"), self.get_node_name("ACTOR"))
        graph.add_edge(self.get_node_name("ACTOR"), self.get_node_name("REPLANNER"))
        graph.add_conditional_edges(
            self.get_node_name("REPLANNER"),
            self._get_should_continue(),
            {
                "end" : END,
                "actor" : self.get_node_name("ACTOR")
            }
        )

    # ------------------------------------------------------------------------------------------------
    # Create Agent Application
    # ------------------------------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        model: BaseChatModel | ChatOpenAI,
        state_schema: type = StateSchema,
        input_schema: type = InputSchema,
        output_shcema: type = OutputSchema,
        agent_name: str = "PlanAndExecuteAgent",
    ) -> "PlanAndExecuteAgent":
        
        self = cls(
            model = model,
            state_schema=state_schema,
            input_schema=input_schema,
            output_shcema=output_shcema,
            agent_name=agent_name
        )

        self.build_graph()

        return self