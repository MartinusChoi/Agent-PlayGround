"""
Reflexion Agent Class
"""

from src.agent.base.base_agent import BaseAgent
from src.prompt.prompt_manager import PromptManager

import os
from typing import Union, ClassVar, Dict, Annotated, List
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    ToolMessage,
    AIMessage,
    HumanMessage
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# -------------------------------------------------------------------------------------
# Define Response Schema
# -------------------------------------------------------------------------------------
class AnswerSchema(BaseModel):
    """
    사용자의 질문에 대해 답변의 생성하기 위해 이 도구을 사용하세요.
    """
    answer: str = Field(description="사용자 질문에 대한 이전 답변과 이전 답변에 대한 비평을 참고하여 개선한 답변")

class EvaluateSchema(BaseModel):
    missing: str = Field(description="사용자의 질문에 대한 현재 답변에서 누락되거나 부족한 부분에 대한 평가")
    superfluous: str = Field(description="사용자의 질문에 대한 현재 답변에서 불필요한 부분에 대한 평가")

class ReflectionSchema(BaseModel):
    reflection: str = Field(description="사용자의 질문에 대한 현재 답변에 대해 평가한 내용을 기반으로한 비평")





# -------------------------------------------------------------------------------------
# Graph State Schema
# -------------------------------------------------------------------------------------
class ReflexionSchema(MessagesState):
    user_query: Annotated[str, "사용자의 질문"]
    trial_num: Annotated[int, "사용자 질문에 대한 답변 생성을 시동한 횟수"]
    trajectory: Annotated[str, "사용자의 질문에 대한 현재 답변"]
    rewards: Annotated[List[str], "사용자의 질문에 대해 답변한 내용을 평가한 결과 리스트"]
    verbal_reflection: Annotated[List[str], "답변을 평가한 내용을 기반으로 작성한 피드백"]





# -------------------------------------------------------------------------------------
# Main Reflextion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflextion Architecture Agent.
    """

    NODE_NAMES: ClassVar[Dict[str, str]] = {
        "INITIAL_ACTOR" : "initial_actor",
        "ACTOR" : "actor",
        "EVALUATOR" : "evaluator",
        "REFLECTOR" : "reflector",
        "INITIAL_SEARCH" : "initial_search",
        "SEARCH" : "search"
    }

    def __init__(
        self,
        model: Union[BaseChatModel, ChatOpenAI],
        state_schema: type,
        input_schema: type,
        output_schema: type,
        agent_name:str
    ) -> None:
        """
        Initialize Reflexion Agent Class

        Args:
            model: Union[BaseChatModel, ChatOpenAI],
            state_schema: type,
            input_schema: type,
            output_schema: type,
            agent_name:str
        """

        super().__init__(
            model=model,
            state_schema=state_schema,
            input_schema=input_schema,
            output_schema=output_schema,
            agent_name=agent_name
        )

        self.prompt_manager = PromptManager(os.path.join('src', 'agent', 'reflexion', 'prompts'))

        search_tool = TavilySearchResults(
            max_results=5,
            search_depth='advanced'
        )
        self.tools = [search_tool]
        

    # -------------------------------------------------------------------------------------
    # Create Agents
    # -------------------------------------------------------------------------------------
    def _get_initial_actor(self) -> Runnable:
        initial_actor_prompt = self.prompt_manager.load_prompt('initial_actor')
        model = self.model.bind_tools(self.tools)
        model = model.with_structured_output(AnswerSchema)
        initial_actor = initial_actor_prompt | model

        return initial_actor
    
    def _get_actor(self) -> Runnable:
        actor_prompt = self.prompt_manager.load_prompt('actor')
        model = self.model.bind_tools(self.tools)
        model = model.with_structured_output(AnswerSchema)
        actor = actor_prompt | model

        return actor
    
    def _get_evaluator(self) -> Runnable:
        evaluator_prompt = self.prompt_manager.load_prompt('evaluator')

        evaluator = evaluator_prompt | self.model.with_structured_output(EvaluateSchema)

        return evaluator
    
    def _get_reflector(self) -> Runnable:
        reflector_prompt = self.prompt_manager.load_prompt('reflector')

        reflector = reflector_prompt | self.model.with_structured_output(ReflectionSchema)

        return reflector

    # -------------------------------------------------------------------------------------
    # Create Nodes
    # -------------------------------------------------------------------------------------
    def _get_initial_actor_node(self):
        initial_actor_agent:Runnable = self._get_initial_actor()

        def initial_actor(state:ReflexionSchema):
            user_query = state['messages'][0].content

            if type(state['messages'][-1]) is ToolMessage:
                response = initial_actor_agent.invoke({
                    "messages" : state["messages"][-2:] + [HumanMessage(content=user_query)]
                })
            else:
                response:AnswerSchema = initial_actor_agent.invoke({
                    "messages" : [HumanMessage(content=user_query)]
                })

                response = AIMessage(content=response.answer)
            
            return {
                "messages" : [response],
                "trial_num" : 0,
                "user_query" : user_query,
                "verbal_reflection" : []
            }

        return initial_actor

    def _get_actor_node(self):
        actor_agent:Runnable = self._get_actor()

        def actor(state:ReflexionSchema):
            
            if type(state['messages'][-1]) is ToolMessage:
                response = actor_agent.invoke({
                    "context" : state["messages"][-2:] + [AIMessage(content=f"""
사용자 최초 질문 : 
{state['user_query']}

질문에 대한 지난 답변 :
{state['trajectory']}

답변에 대한 피드백 : 
{state['verbal_reflection']}
""")]
                })
            else:
                response:AnswerSchema = actor_agent.invoke({
                    "context" : [AIMessage(content=f"""
사용자 최초 질문 : 
{state['user_query']}

질문에 대한 지난 답변 :
{state['trajectory']}

답변에 대한 피드백 : 
{'  ,  '.join(state['verbal_reflection'])}
""")]
                })
            
                response = AIMessage(content=response.answer)
            
            return {"messages" : [response]}

        return actor
    
    def _get_evaluator_node(self):
        evaluator_agent:Runnable = self._get_evaluator()

        def evaluator(state:ReflexionSchema):
            response = evaluator_agent.invoke({"evaluate_request" : [HumanMessage(content=f"""
최초 사용자 질문 : 
{state['user_query']}
                                                           
질문에 대한 현재 답변 :
{state['messages'][-1].content}
                                                
질문에 대한 현재 답변을 평가해주세요.
""")]})
    
            new_state = state.copy()
            new_state["rewards"] = [response.missing, response.superfluous]
            new_state["trajectory"] = state['messages'][-1].content
            new_state["trial_num"] += 1
            new_state["verbal_reflection"]

            return new_state
        
        return evaluator
    
    def _get_reflection_node(self):
        reflector_agent = self._get_reflector()

        def reflector(state: ReflexionSchema):
            response = reflector_agent.invoke({"reflection_request" : [HumanMessage(content=f"""
최초 사용자 질문 : 
{state['user_query']}
                                                           
질문에 대한 현재 답변 :
{state['trajectory']}
                                                           
답변에 대한 평가 자료:
1. 답변에서 누락되거나 부족한 부분 :
{state['rewards'][0]}
2. 답변에서 불필요한 부분 :
{state['rewards'][1]}
                                                
질문에 대한 현재 답변을 평가해주세요.
""")]})
    
            new_state = state.copy()
            if new_state["verbal_reflection"] : 
                new_state['verbal_reflection'].append(response.reflection)
                if len(new_state['verbal_reflection']) > 3 : new_state['verbal_reflection'] = new_state['verbal_reflection'][-3:]
            else : new_state["verbal_reflection"] = [response.reflection]

            return new_state

        return reflector
    
    def _get_initial_tool_node(self):
        return ToolNode(tools=self.tools, name="initial_search")
    
    def _get_tool_node(self):
        return ToolNode(tools=self.tools, name="search")
    
    # -------------------------------------------------------------------------------------
    # Create Conditional Edges
    # -------------------------------------------------------------------------------------
    def _get_initial_should_continue(self):

        def initial_should_continue(state: ReflexionSchema):
            ai_message = state['messages'][-1]

            # print(ai_message)

            if hasattr(ai_message, "tool_calls") and (ai_message.tool_calls) and (ai_message.tool_calls[0]["name"] == "tavily_search_results_json") : return "search"
            else : return "evaluate"
        
        return initial_should_continue
    
    def _get_should_continue(self):

        def should_continue(state: ReflexionSchema):
            ai_message = state['messages'][-1]

            if hasattr(ai_message, "tool_calls") and (ai_message.tool_calls) and (ai_message.tool_calls[0]["name"] == "tavily_search_results_json") : return "search"
            elif state['trial_num'] > 3 : return "end"
            else : return "evaluate"
        
        return should_continue


    # -------------------------------------------------------------------------------------
    # Agent Core Logic (Initialize Nodes, Edges)
    # -------------------------------------------------------------------------------------
    def _init_nodes(self, graph:StateGraph) -> None:
        graph.add_node(self._get_initial_actor_node())
        graph.add_node(self._get_actor_node())
        graph.add_node(self._get_evaluator_node())
        graph.add_node(self._get_reflection_node())
        graph.add_node(self._get_initial_tool_node())
        graph.add_node(self._get_tool_node())

    def _init_edges(self, graph:StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("INITIAL_ACTOR"))
        graph.add_conditional_edges(
            self.get_node_name('INITIAL_ACTOR'),
            self._get_initial_should_continue(),
            {
                "search" : self.get_node_name("INITIAL_SEARCH"),
                "evaluate" : self.get_node_name("EVALUATOR")
            }
        )
        graph.add_edge(self.get_node_name("INITIAL_SEARCH"), self.get_node_name("INITIAL_ACTOR"))
        graph.add_edge(self.get_node_name("EVALUATOR"), self.get_node_name("REFLECTOR"))
        graph.add_edge(self.get_node_name("REFLECTOR"), self.get_node_name("ACTOR"))
        graph.add_conditional_edges(
            self.get_node_name("ACTOR"),
            self._get_should_continue(),
            {
                "search" : self.get_node_name("SEARCH"),
                "end" : END,
                "evaluate" : self.get_node_name("EVALUATOR")
            }
        )
        graph.add_edge(self.get_node_name("SEARCH"), self.get_node_name("ACTOR"))
    
    # -------------------------------------------------------------------------------------
    # Create Graph Application
    # -------------------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        model: BaseChatModel | ChatOpenAI,
        state_schema: type = ReflexionSchema,
        input_schema: type = None,
        output_schema: type = None,
        agent_name: str = "ReflextionAgent"
    ) -> Runnable:
        self = cls(
            model = model,
            state_schema=state_schema,
            input_schema=input_schema,
            output_schema=output_schema,
            agent_name=agent_name
        )

        self.build_graph()

        return self