from src.agent.reflexion.reflexion_agent import ReflexionAgent
from src.utils.run import stream
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os



load_dotenv(os.path.join('..', '..', 'config', '.env'))

agent = ReflexionAgent.create(
    model=ChatOpenAI(
        model='gpt-4o',
        temperature=0.0
    )
)

stream(agent.graph, {
    "messages" : [HumanMessage(content="2025년 9월 현재 기준, 최근 LLM 기술에 대한 동향에 대해 정리해줘")]
})