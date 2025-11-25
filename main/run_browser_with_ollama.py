import os
import asyncio
from browser_use import Agent
from browser_use.llm.ollama.chat import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("BROWSER_USE_LLM_MODEL", "deepseek-r1:32b"),  # gpt-oss:20b    deepseek-r1:32b
    host=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
)

async def run_agent():
    agent = Agent(
        # task="find the founders of browser-use",
        # task="get latest price of xauusd. but also try to use different sources if 3 times the same source is used",
        task="open this page: 'https://www.tradingview.com/chart/?symbol=ECONOMICS:FRGDPYY' and figure out if this link has finance information.",
        llm=llm,
        max_actions_per_step=5
    )
    result = await agent.run(max_steps=15)
    print(result)

if __name__ == "__main__":
    asyncio.run(run_agent())
