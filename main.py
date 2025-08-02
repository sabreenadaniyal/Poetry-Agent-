from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, trace
from dotenv import load_dotenv
import os
import asyncio

# 🌱 Load environment variables
load_dotenv()

# 🔐 Get API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 🤖 External Client Setup
external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🧠 Model Configuration
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

# ⚙️ Run Configuration
config = RunConfig(
    model = model,
    model_provider = external_client,
    # tracing_disabled = True
)

################ 🎵 ----------------------------- LYRIC AGENT ---------------------------------- ################
lyric_agent = Agent(
    name = "Lyric Analyst",
    instructions = 
    """
    You are a lyric analyst. Analyze the poem and explain whether it expresses deep personal
    emotions and feelings.
    """
)

################ 📖 ----------------------------- NARRATIVE AGENT -------------------------------- ###############
narrative_agent = Agent(
    name = "Narrative Analyst",
    instructions = 
    """
    You are a narrative analyst. Analyze the poem and determine if it contains a storyline 
    or personal experience that conveys deep emotion.
    """
)

################ 🎭 ---------------------------- DRAMATIC AGENT ---------------------------------- ##############
dramatic_agent = Agent(
    name = "Dramatic Analyst",
    instructions = 
    """
    You are a dramatic analyst. Assess the poem for theatrical expression and emotional intensity.
    """
)

# 👨‍🏫 Parent Agent Setup
parent_agent = Agent(
    name = "Parent Agent",
    instructions = 
    """
    You are a poetry classification agent. Based on the input poem, hand off the analysis to
    the most appropriate specialist: Lyric, Narrative, or Dramatic.
   """,
   handoffs = [lyric_agent, narrative_agent, dramatic_agent]
)

# 🚀 Main Function
async def main():
    with trace("poetry agent"):
     poem = """
    I walk alone beneath the sky so wide,
    My thoughts adrift like waves upon the tide.
    The world moves on, yet I remain,
    A heart once whole now split by pain.
    """
    result = await Runner.run(
        parent_agent,
        input = poem,
        run_config = config
    )

     # 🎉 Print Results with color
    print("\033[95m--- 🎯 Final Output ---\033[0m")
    print(f"\033[93m{result.final_output}\033[0m")

    print("\033[94m--- 🎭 Last Agent ---\033[0m")
    print(f"\033[96mLast Agent ==>\033[0m {result.last_agent.name}")
    
# 🏁 Run
if __name__ == "__main__":
    asyncio.run(main())