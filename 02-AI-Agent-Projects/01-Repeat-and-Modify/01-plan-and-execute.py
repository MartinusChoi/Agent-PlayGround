import os
from dotenv import load_dotenv

from src.agent.plan_and_execute.plan_and_execute_agent import PlanAndExecuteAgent
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join("..", "..", "config", ".env"))

agent = PlanAndExecuteAgent.create(
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
)

request = "2025년 9월 기준 OpenAI 관련 최신 뉴스를 정리해줘"

for event in agent.graph.stream({"request":request}):
    for node, state in event.items():
        print("======" * 30)
        print(f"Node : {node}")
        for key, value in state.items():
            print(key)
            print(f"{value}\n")
        print("======" * 30, "\n")