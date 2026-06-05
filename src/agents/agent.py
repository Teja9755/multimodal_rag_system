from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.retrieval.retrieval import query_documents

load_dotenv()

# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# =========================================================
# RETRIEVAL NODE
# =========================================================

def retrieval_node(state: AgentState):

    user_question = state["messages"][-1].content

    docs = query_documents(
        query=user_question,
        k=5
    )

    context = "\n\n".join(
        d["content"]
        for d in docs
    )

    return {
        "retrieved_context": context
    }

# =========================================================
# AGENT NODE
# =========================================================

def agent_node(state):

    question = state["messages"][-1].content

    docs = query_documents(
        query=question,
        k=5
    )

    context = "\n\n".join(
        doc["content"]
        for doc in docs
    )

    system_prompt = SystemMessage(
        content=f"""
You are a Financial Document Assistant.

Use ONLY the retrieved context.

If answer exists in context:
    Answer clearly.

If answer is not present:
    Say:
    "I could not find that information in the documents."

Retrieved Context:
{context}
"""
    )

    response = llm.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response]
    }

# =========================================================
# BUILD GRAPH
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)

workflow.add_edge(
    START,
    "agent"
)

workflow.add_edge(
    "agent",
    END
)

app = workflow.compile()

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the gross revenue for FY25?"
                )
            ]
        }
    )

    print("\nFINAL ANSWER\n")
    print(result["messages"][-1].content)