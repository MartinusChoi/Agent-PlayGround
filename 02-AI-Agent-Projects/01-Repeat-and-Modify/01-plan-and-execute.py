from src.agent.plan_and_execute.plan_and_execute_agent import PlanAndExecuteAgent
from src.utils.run import stream
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI



load_dotenv(os.path.join("..", "..", "config", ".env"))

agent = PlanAndExecuteAgent.create(
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
)

stream(agent.graph, {"request" : "2025년 9월 기준 OpenAI 관련 최신 뉴스를 정리해줘"})