from fastapi import FastAPI, Request
from crewai import Agent, Task, Crew
import os

app = FastAPI()

# 환경 변수에서 API 키 불러오기
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 1. 에이전트 정의
triage_agent = Agent(
    name="TriageAgent",
    role="우선순위 분류자",
    goal="요청의 긴급도를 판단합니다.",
    prompt="You are a ticket triage assistant. Classify the urgency."
)

responder_agent = Agent(
    name="ResponderAgent",
    role="응답 생성자",
    goal="정중하고 적절한 응답을 생성합니다.",
    prompt="You are a support responder. Create a helpful response."
)

logger_agent = Agent(
    name="LoggerAgent",
    role="로그 기록자",
    goal="처리 내용을 기록합니다.",
    prompt="You are a logging assistant. Record logs."
)

# 2. FastAPI 라우트 정의
@app.post("/agent")
async def run_agents(req: Request):
    data = await req.json()
    description = data.get("description", "요청 없음")

    task1 = Task(description="고객 요청 분류 및 응답 생성", agent=triage_agent, input=description)
    task2 = Task(description="응답을 로그로 기록", agent=logger_agent, context=[task1])

    crew = Crew(agents=[triage_agent, responder_agent, logger_agent], tasks=[task1, task2])
    result = crew.kickoff()
    return {"result": result}
