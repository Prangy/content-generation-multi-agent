from langchain_openai import ChatOpenAI
from baseagent import BaseAgent
from blogsearcheragent import BlogSearcherAgent
from contentwriteragent import ContentWriterAgent
from imagequalityagent import ImageQualityAgent
from imagesearcheragent import ImageCreatorAgent
from videosearcheragent import VideoSearcherAgent
from websearcheragent import WebSearcherAgent
from webSearcherQualityAgent import webSearcherQualityAgent
from BlogSearcherQualityAgent import BlogSearcherQualityAgent
# Agent Factory
class AgentFactory:
    agents = {
        "Web_Searcher": WebSearcherAgent,
        "web_Searcher_Quality" : webSearcherQualityAgent,
        "Image_Generator": ImageCreatorAgent,
        "Image_Quality_Agent": ImageQualityAgent,
        "Blog_Searcher": BlogSearcherAgent,
        "blog_searcher_quality": BlogSearcherQualityAgent,
        "Content_Writer": ContentWriterAgent         
    }

    @staticmethod
    def get_agent(agent_type: str, llm: ChatOpenAI, tools: list, system_prompt: str) -> BaseAgent:
        agent_class = AgentFactory.agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type {agent_type} not recognized.")
        return agent_class(llm, tools, system_prompt)
