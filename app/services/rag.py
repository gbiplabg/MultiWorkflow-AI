
import os
import faiss

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm= ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv('GROQ_API_KEY'))

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

embedding_dim = len(embeddings.embed_query("sample text"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


def load_and_split_pdf_dynamic(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
    """Load a PDF and split into chunks."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No pages could be extracted from the PDF.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)
    except Exception as e:
        raise RuntimeError(f"Error while loading and splitting the PDF: {str(e)}")


@tool
def retrieve_context(query: str):
    """Retrieve clean context from vector store for answering queries."""
    try:
        retrieved_docs = vector_store.similarity_search(query, k=2)
        if not retrieved_docs:
            return "No relevant context found."
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    except Exception as e:
        return f"Error while retrieving context: {str(e)}"


tools = [retrieve_context]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)




def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                SystemMessage(
                    content=(
                       """You are a Retrieval-Augmented Generation (RAG) assistant. 
                        Always query the vector database first before generating a response. 
                        Follow these rules:

                        1. Retrieve the most relevant context chunks from the database (top-k search).
                        2. If relevant content is found, use it as the primary source. Do not fabricate details.
                        3. If multiple documents are retrieved, compare and summarize them clearly.
                        4. If no relevant information is found:
                        - For knowledge-based questions → respond exactly with: 
                            "I could not find relevant information in the knowledge base."
                        - For casual conversation (e.g., greetings, chit-chat, generic questions) → reply naturally using your internal knowledge.
                        5. Always be clear, concise, and directly address the user query.

                        Your role: Act as a knowledge-grounded assistant while remaining conversational when appropriate.
                    """
                    )
)
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue or stop, with error handling."""
    try:
        last_message = state["messages"][-1]
        return "tool_node" if getattr(last_message, "tool_calls", None) else END
    except Exception as e:
        print(f"[should_continue] Error: {e}")
        return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

