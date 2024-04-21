import functools
from typing import Callable
from baseagent import BaseAgent
from agent_node import agent_node

class webSearcherQualityAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)
 