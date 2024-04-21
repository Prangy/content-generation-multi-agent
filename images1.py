import functools
from bs4 import BeautifulSoup
import functools
from typing import Callable, Dict, Any, Sequence
import operator
import os
from agentfactory1 import AgentFactory
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from serpapi import GoogleSearch
from langchain.agents import load_tools
from langchain.agents import initialize_agent, load_tools

from langchain_core.tools import tool
import requests

# Assuming necessary imports and environment setup as in your initial script
os.environ["OPENAI_API_KEY"] = "sk-proj-********************************"
#os.environ["OPENAI_API_KEY"] = "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"

# os.environ['TAVILY_API_KEY'] ="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"
os.environ['TAVILY_API_KEY']="tvly-****************************"
os.environ['SERPAPI_API_KEY'] ="************************************"
#os.environ["LANGCHAIN_API_KEY"] = "ls__66140278b9c44003890a3aa152caf849"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_PROJECT"] = "imgGen"

llm = ChatOpenAI(model="gpt-4-turbo-preview")

#image tool
image_generate_tool = load_tools(["dalle-image-generator"])

#search tool
tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

# Agent State and Workflow Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
# Create the workflow using the Factory pattern
# Create the workflow using the Factory pattern
def create_workflow(llm, tools, image_generate_tool):
    
    Image_Creator_system_prompt = "You are an image generator. Generate the images for the given information."
    image_quality_agent_system_prompt = "You are a image Quality Agent. Analysis the image searcher agent work and provide the relevence, engagement, image quality and helpfullness on a scale of 1 to 100 in a tabular form of the provided content and if the score is below 90 then ask the Image_Creator to again generate."

    image_creator_agent = AgentFactory.get_agent("Image_Generator", llm, image_generate_tool, Image_Creator_system_prompt)
    image_quality_agent = AgentFactory.get_agent("Image_Quality_Agent", llm, tools, image_quality_agent_system_prompt)

    Image_Creator_node = image_creator_agent.create_node("Image_Generator")
    image_quality_node = image_quality_agent.create_node("Image_Quality_Agent")

    members = ["Image_Generator", "Image_Quality_Agent"]
    system_prompt = (
        """As a supervisor, your role is to oversee a dialogue between these
        workers: {members}. and excute all the agents one by one,
        determine which worker should take the next action. Each worker is responsible for
        executing a specific task and reporting back their findings and progress. Once all tasks are complete,
        indicate with 'FINISH'."""
    )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
    ]).partial(options=str(options), members=", ".join(members))

    manager_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())
    
    # Note: This function setup should end with workflow setup and potentially running the workflow,
    # similar to your original script's structure.
    workflow = StateGraph(AgentState)
    workflow.add_node("Image_Generator", Image_Creator_node)
    workflow.add_node("Image_Quality_Agent", image_quality_node)
    workflow.add_node("manager", manager_chain)

    # Adjust the edges to reflect the desired order of execution
    for member in members:
        workflow.add_edge(member, "manager")


    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    conditional_map["manager"] = "Image_Generator"
    workflow.add_conditional_edges("manager", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("manager")

    graph = workflow.compile()

    # Run the graph
    for s in graph.stream({
        "messages": [HumanMessage(content="""Generate images for the latest AI trends in 2020 and assess their quality.""")]
    }):
        if "__end__" not in s:
            print(s)
            print("----")

# Example call to setup and potentially run the workflow
# Assuming 'tools' is defined and initialized as in your script
create_workflow(llm, tools, image_generate_tool)

# """Search information for the latest AI trends in 2020 with the use of web searcher,
#                 summarize the content. After summarise pass it on to image searcher, Image_Quality_Agent, video searcher and blog searcher
#                 one by one to fetch the url of image, video and blog or article for each trend. After fetch the url of
#                 images, videos, and blogs pass it on to content writer for processing and combine them"""
