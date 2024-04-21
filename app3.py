import functools
from bs4 import BeautifulSoup
import functools
from typing import Callable, Dict, Any, Sequence
import operator
import os
from agentfactory import AgentFactory
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
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

from langchain_core.tools import tool
import requests


# Assuming necessary imports and environment setup as in your initial script
os.environ["OPENAI_API_KEY"] = "sk-proj-sLnQKnmXnD6Wp3aTpDqkT3BlbkFJDZKkyqFpeMNQdraHLH8d"
os.environ['TAVILY_API_KEY'] ="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"
os.environ['SERPAPI_API_KEY'] ="2f6cda02b12ababf9315904ec550440ab79c28fc4a467bc14e5ee98c258a88e4"

llm = ChatOpenAI(model="gpt-4-turbo-preview")

tavily_tool = TavilySearchResults(max_results=5)


#Image Searh tool for search the image urls
@tool
def image_search_tool(query: str, api_key: str,num_results: int = 5):
    """
    This function interacts with the Google Search API to retrieve search results based on the provided query.

    Args:
        query (str): The search query.
        api_key (str): The API key for accessing the Google Search API.

    Returns:
        dict: The search results obtained from the Google Search API.
    """
    params = {
        "engine": "google_images",
        "q": query,
        "num": num_results,
        "api_key": "6655aaaf86da7c57f3a789c10953507757b71b76a29d1b4cabba1ae2493f0442"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    #Truncate the results to the desired number
    truncated_results = {k: v[:num_results] for k, v in results.items() if isinstance(v, list)}
    
    return truncated_results

#Video Search tool for search the video urls
@tool
def video_search_tool(query: str, api_key: str):
    """
    This function interacts with the Google Search API to retrieve search results based on the provided query.

    Args:
        query (str): The search query.
        api_key (str): The API key for accessing the Google Search API.

    Returns:
        dict: The search results obtained from the Google Search API.
    """
    params = {
        "engine": "google_videos",
        "q": query,
        "api_key": "6655aaaf86da7c57f3a789c10953507757b71b76a29d1b4cabba1ae2493f0442"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results


#Blog Search tool from search the blog urls
@tool
def blog_search_tool(query: str, api_key: str):
    """
    This function interacts with the Google Search API to retrieve search results based on the provided query.

    Args:
        query (str): The search query.
        api_key (str): The API key for accessing the Google Search API.

    Returns:
        dict: The search results obtained from the Google Search API.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": "6655aaaf86da7c57f3a789c10953507757b71b76a29d1b4cabba1ae2493f0442"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

tools = [tavily_tool]
#image_generate_tool = load_tools(["dalle-image-generator"])

image_tool = [image_search_tool]

#video_tool = [video_search_tool]

blog_tool = [blog_search_tool]

# Agent State and Workflow Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create the workflow using the Factory pattern
def create_workflow(llm: ChatOpenAI, tools: list, image_tool: list,  blog_tool: list):
    
    web_searcher_system_prompt = "You are a web searcher. Search the internet for information."
    web_Searcher_Quality_system_prompt = "You are a web searcher Quality Agent. Analysis the web searcher agent work and provide the relevance, grammatical correctness, engagement, harmfulness, and helpfulness on a scale of 1 to 100 in a tabular form of the provided content. If the score is below 90, then ask the web searcher to generate again."
    blog_searcher_system_prompt = "You are a blog searcher. Fetch the url of the blogs or articles for the given information."
    blog_searcher_quality_system_prompt = "You are a blog searcher quality agent. Your task is to assess the quality of the blog searcher agent for each provided blog or article URL. Analyze the content and provide ratings on a scale of 1 to 100 for relevance, grammatical correctness, engagement, harmfulness, and helpfulness. Present your evaluations in a tabular form, detailing the scores for each aspect alongside the corresponding URL. If the score is below 90, then ask the blog searcher to generate again."
    image_generator_system_promt = "You are an image searcher. Search and fetch the url  of the images for the given information ."
    image_generator_quality_system_prompt = "You are a image Quality Agent. Analysis the image searcher  agent work and provide the relevence, engagement, image quality and helpfullness on a scale of 1 to 100 in a tabular form of the provided content and if the score is below 90 then ask the Image_Creator to again generate."
    content_writer_system_prompt = "You are a content writer. Process the output of all the others agents and combine them."
    
    # Initialize agents using the factory
    web_searcher_agent = AgentFactory.get_agent("Web_Searcher", llm, tools, web_searcher_system_prompt)
    web_Searcher_Quality_agent = AgentFactory.get_agent("web_Searcher_Quality",llm, tools, web_Searcher_Quality_system_prompt)
    blog_searcher_agent = AgentFactory.get_agent("Blog_Searcher", llm, blog_tool, blog_searcher_system_prompt)
    blog_searcher_quality_agent = AgentFactory.get_agent("blog_searcher_quality", llm, tools, blog_searcher_quality_system_prompt)
    image_searcher_agent = AgentFactory.get_agent("Image_Generator", llm, image_tool, image_generator_system_promt)
    image_quality_agent = AgentFactory.get_agent("Image_Quality_Agent", llm, tools, image_generator_quality_system_prompt)
    #video_searcher_agent = AgentFactory.get_agent("Video_Searcher", llm, video_tool, video_searcher_system_prompt)
    content_writer_agent = AgentFactory.get_agent("Content_Writer", llm, tools, content_writer_system_prompt)

    # Create nodes
    web_searcher_node = web_searcher_agent.create_node("Web_Searcher")
    web_Searcher_Quality_node = web_Searcher_Quality_agent.create_node("web_Searcher_Quality")
    blog_searcher_node = blog_searcher_agent.create_node("Blog_Searcher")
    blog_searcher_quality_node = blog_searcher_quality_agent.create_node("blog_searcher_quality")
    image_searcher_node = image_searcher_agent.create_node("Image_Generator")
    image_quality_node = image_quality_agent.create_node("Image_Quality_Agent")
    #video_searcher_node = video_searcher_agent.create_node("Video_Searcher")
    content_writer_node = content_writer_agent.create_node("Content_Writer")

    # Supervisor and Workflow Logic as in your original script, adapted for the new setup
    # Setup your supervisor, conditional logic, and integrate the nodes into the workflow
    members = ["Web_Searcher", "web_Searcher_Quality", "Blog_Searcher", "blog_searcher_quality", "Image_Generator", "Image_Quality_Agent", "Content_Writer"]
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

    workflow.add_node("Web_Searcher", web_searcher_node)
    workflow.add_node("web_Searcher_Quality", web_Searcher_Quality_node)
    workflow.add_node("Blog_Searcher", blog_searcher_node)
    workflow.add_node("blog_searcher_quality", blog_searcher_quality_node)
    workflow.add_node("Image_Generator", image_searcher_node)
    workflow.add_node("Image_Quality_Agent", image_quality_node)
    #workflow.add_node("Video_Searcher", video_searcher_node)
    workflow.add_node("Content_Writer", content_writer_node)
    workflow.add_node("manager", manager_chain)

    for member in members:
        workflow.add_edge(member, "manager")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("manager", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("manager")

    graph = workflow.compile()


    # Run the graph
    for s in graph.stream({
        "messages": [HumanMessage(content="""Search information for the latest AI trends in 2024 with the use of web searcher,
                summarize the content. After summarise pass it on to image searcher, Image_Quality_Agent, video searcher and blog searcher
                one by one to fetch the url of image, video and blog or article for each trend. After fetch the url of
                images, videos, and blogs pass it on to content writer for processing and combine them""")]
    }, {"recursion_limit": 100}):
        if "__end__" not in s:
            print(s)
            print("----")
        if "Content_writer" in s:
            agent_content = s["Content_Writer"]["messages"]
            print(agent_content,'content')
            GeneratedForecast = agent_content[0].content
            print("generated_forecast-----------------------------",GeneratedForecast,'generated_forecast------------------------')
            return GeneratedForecast

@app.route('/content', methods=['POST'])
def initiate_api():
    data = request.get_json()
    user_input = data.get('user_input')
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    if not user_input:
        return jsonify({"error": "User Input is required"}), 400
    
    try:
        
        image_tool = [image_search_tool]

        #video_tool = [video_search_tool]

        blog_tool = [blog_search_tool]
        generated_forecast = create_workflow(llm, tools, image_tool,  blog_tool)

        print(generated_forecast,'last_response')
        if generated_forecast:
             return jsonify({"message": "Forecast generated", "response": generated_forecast}), 200
        else:
            return jsonify({"error": "Content Not Generated Try again"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9000, host="0.0.0.0")


# create_workflow(llm, tools, image_tool,  blog_tool)