from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


from dotenv import load_dotenv

load_dotenv()

''' Normal one without using the prompt template'''

# prompt template

def genereate_lion_name(animal_type, animal_color):
    llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.11
        )
    prompt_template_name = PromptTemplate(
        input_variables = ['animal_type','animal_color'],    
        template="I have a {animal_type} and i want a cool name for it, can you suggest me a good name for my pet. and response should be only of names like four or five names nothing else, like no extra information. It is {animal_color} color"
    )
    name_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_name,
        output_key="animal_name"
    )
    response = name_chain.invoke({
        'animal_type': animal_type,
        'animal_color': animal_color                          
    })
    return response["text"]

# Langchain agent

def langchain_agent():

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and return the product."""
        return a * b

    # wikipedia = WikipediaQueryRun(
    #     api_wrapper=WikipediaAPIWrapper()
    # )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.11
    )

    agent = create_agent(
        model = llm,
        tools = [multiply],
    )

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is a typical average lifespan for a lion in the wild? Then multiply that estimate by 20. State that the lifespan is an estimate."
                    "and the result."
                ),
            }
        ]
    })

    return result["messages"][-1].content

print(langchain_agent())
# print(genereate_lion_name("tiger","orange"))