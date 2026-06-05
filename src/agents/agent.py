import os
import json
import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


from src.api.schema.query_schema import AIResponse
from src.api.tools.tools import RAGState, vector_search_node


load_dotenv()




# ── Helper: build the OpenAI LLM ──────────────────────────────────────────────


def _get_llm() -> ChatOpenAI:
   return ChatOpenAI(
       model=os.getenv("OPENAI_CHAT_MODEL"),
       api_key=os.getenv("OPENAI_API_KEY")
   )






# ── Node 1: Rerank ──────────────────────────────────────────────────────────────
# Uses Cohere's cross-encoder reranker.
# Unlike bi-encoders (which embed query and doc separately),
# a cross-encoder sees query + doc TOGETHER → more accurate relevance scoring.


def rerank_node(state: RAGState) -> RAGState:
   co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
   docs = state["retrieved_docs"]


   rerank_response = co.rerank(
       model="rerank-v3.5",
       query=state["query"],
       documents=[doc.page_content for doc in docs],
       top_n=10
   )


   # Map Cohere result indices back to LangChain Document objects
   reranked_docs = [docs[r.index] for r in rerank_response.results]


   print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
   for i, r in enumerate(rerank_response.results):
       print(f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}")


   return {**state, "reranked_docs": reranked_docs}


# ── Node 2: Generate Answer ─────────────────────────────────────────────────
# Formats the top 10 reranked chunks as context and calls the LLM.
# Uses structured output to enforce the AIResponse schema.
def generate_answer_node(state: RAGState) -> RAGState:
   llm = _get_llm()
   structured_llm = llm.with_structured_output(AIResponse)


   context = "\n\n".join([
       f"""[Source: {doc.metadata.get('source', 'unknown')} | Page:
           {doc.metadata.get('page', -1) +
           1 if doc.metadata.get('page') is not None else '?'}]\n{doc.page_content}"""
       for doc in state["reranked_docs"]
   ])


   prompt = ChatPromptTemplate.from_messages([
       (
           "system",
           "You are a helpful assistant. Answer the user's question using only the "
           "provided context.\n\n"
           "IMPORTANT: The context may contain chunks from MULTIPLE versions of the same "
           "document (e.g. a 2025 edition and a 2026 edition). When the answer differs "
           "across versions, do NOT pick only one. Instead:\n"
           "  - Lead with the most recent / current version's answer (highest year).\n"
           "  - Then explicitly note how earlier versions differed "
           "(e.g. 'As of the 2026 policy ...; previously, under the 2025 policy ...').\n"
           "  - If all versions agree, just give the single answer.\n\n"
           "Citation rules (fill the structured fields):\n"
           "  - document_name: comma-separated list of EVERY source document you used.\n"
           "  - page_no: comma-separated page numbers, aligned with the documents above.\n"
           "  - policy_citations: a readable citation combining each document and its page "
           "(e.g. 'HR_Knowledge_Base_2026.pdf, Page 1; HR_Knowledge_Base_2025.pdf, Page 1').\n"
           "Always cite ALL versions you drew the answer from, not just one."
       ),
       ("human", "Context:\n{context}\n\nQuestion: {query}")
   ])


   chain = prompt | structured_llm
   result = chain.invoke({"context": context, "query": state["query"]})


   print(f"[generate_answer_node] Answer generated.")
   return {**state, "response": result.model_dump()}




# ── Build the LangGraph ────────────────────────────────────────────────────────
# Linear document-RAG pipeline:
#
#   vector_search ──► rerank ──► generate_answer ──► END


def build_rag_graph():
   graph = StateGraph(RAGState)


   graph.add_node("vector_search", vector_search_node)
   graph.add_node("rerank", rerank_node)
   graph.add_node("generate_answer", generate_answer_node)


   graph.set_entry_point("vector_search")


   graph.add_edge("vector_search", "rerank")
   graph.add_edge("rerank", "generate_answer")
   graph.add_edge("generate_answer", END)


   compiled_agent = graph.compile()
 


   return compiled_agent






# Compile once at module load — reused across all requests
rag_graph = build_rag_graph()




# ── Public entrypoint (called by query_service.py) ─────────────────────────
def run_search_agent(query: str) -> dict:
   initial_state: RAGState = {
       "query": query,
       "retrieved_docs": [],
       "reranked_docs": [],
       "response": {},
   }
   final_state = rag_graph.invoke(initial_state)
   return final_state["response"]