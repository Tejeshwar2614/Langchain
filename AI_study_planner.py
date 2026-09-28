import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


@tool
def calculate_topic_time(topics: str, days: int, hours_per_day: float) -> str:
    """Calculate how many study minutes to allocate to each topic."""
    topic_list = [topic.strip() for topic in topics.split(",") if topic.strip()]

    if not topic_list:
        return json.dumps({"error": "Please provide at least one topic."})

    if days <= 0 or hours_per_day <= 0:
        return json.dumps({"error": "Days and study hours must be greater than zero."})

    total_minutes = int(days * hours_per_day * 60)
    minutes_per_topic = total_minutes // len(topic_list)

    allocation = {
        "topics": topic_list,
        "days": days,
        "minutes_per_day": int(hours_per_day * 60),
        "minutes_per_topic": minutes_per_topic,
    }

    return json.dumps(allocation)


@tool
def create_study_plan(allocation_json: str) -> str:
    """Create a day-by-day study plan from the calculator tool's JSON result."""
    try:
        allocation = json.loads(allocation_json)
        topics = allocation["topics"]
        days = allocation["days"]
        minutes_per_day = allocation["minutes_per_day"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "I couldn't read the time allocation. Please try again with topics, days, and hours per day."

    if not topics or days <= 0:
        return "The plan needs at least one topic and one study day."

    plan = []

    for day in range(1, days + 1):
        # Rotate the starting topic each day for variety.
        start = (day - 1) % len(topics)
        daily_topics = topics[start:] + topics[:start]

        # Split the day's study minutes across the topics.
        base_minutes = minutes_per_day // len(topics)
        extra_minutes = minutes_per_day % len(topics)

        sessions = []
        for index, topic in enumerate(daily_topics):
            minutes = base_minutes + (1 if index < extra_minutes else 0)
            if minutes:
                sessions.append(f"{topic}: {minutes} minutes")

        plan.append(f"Day {day}: " + "; ".join(sessions))

    return "\n".join(plan)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)

agent = create_agent(
    model=llm,
    tools=[calculate_topic_time, create_study_plan],
    system_prompt=(
        "You are a study planner. When a user requests a plan, first call "
        "calculate_topic_time with their topics, number of days, and study hours "
        "per day. Then pass that tool's complete JSON result to create_study_plan. "
        "Use the study plan tool's result in your final answer. If the user omits "
        "any required detail, ask them for it."
    ),
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Make me a study plan for Python, LangChain, and math. "
                "I have days and can study hours per day."
            ),
        }
    ]
})

print(result["messages"][-1].content)