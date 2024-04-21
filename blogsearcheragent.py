import functools
from typing import Callable
from agent_node import agent_node
from baseagent import BaseAgent


class BlogSearcherAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)