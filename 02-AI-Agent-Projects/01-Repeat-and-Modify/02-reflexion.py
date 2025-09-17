from src.agent.reflexion.reflexion_agent import ReflexionAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

load_dotenv(os.path.join('..', '..', 'config', '.env'))

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0
)

agent = ReflexionAgent.create(model=llm)

for event in agent.graph.stream({"messages" : [HumanMessage(content="2025년 9월 현재 기준, 최근 LLM 기술에 대한 동향에 대해 정리해줘")]}):
    for node_name, state in event.items():
        print("=============", f"[Node : '{node_name}']", "=============")
        for key, value in state.items():
            print(f"    {key} : \n      {value}")
    print("\n", "============"*15, "\n")