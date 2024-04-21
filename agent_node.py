from langchain_core.messages import BaseMessage, HumanMessage

# Helper functions
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}