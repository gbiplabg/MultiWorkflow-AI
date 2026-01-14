import os
import uuid
import shutil


from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.schema import AIMessage

from services.rag import vector_store, UPLOAD_DIR, load_and_split_pdf_dynamic, agent
from services.flow_graph import flow_graph


app = FastAPI(title="INO AI")
templates = Jinja2Templates(directory="templates")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions = {}     
flow_sessions = {}       


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the frontend UI"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """Upload a PDF and store its embeddings."""
    try:
        if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are allowed."}
            )

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
     
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = load_and_split_pdf_dynamic(file_path)
        if not docs:
            return JSONResponse(
                status_code=400,
                content={"error": "Unabe to extract the text from the PDF."}
            )
        vector_store.add_documents(docs)

        return JSONResponse(
            {"message": "PDF uploaded and processed successfully!", "file_id": file_id}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process PDF: {str(e)}"}
        )

@app.post("/chat/")
async def chat_with_pdf(query: str = Form(...), user_id: str = Form("default_user")):
    """Chat with the uploaded PDFs."""
    try:
        if user_id not in chat_sessions:
            chat_sessions[user_id] = str(uuid.uuid4())

        configurable = {"thread_id": chat_sessions[user_id]}
        messages = [HumanMessage(content=query)]
        
        response = agent.invoke({"messages": messages}, configurable)
        print("Chat Response:", response)

        replies = [m.content for m in response["messages"] if isinstance(m, AIMessage)]
        final_reply = replies[-1] if replies else "No response."

        return {"response": final_reply, "thread_id": chat_sessions[user_id]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {str(e)}"})


class ChatRequest(BaseModel):
    user_id: str = "default_user"
    user_message: str


@app.post("/chat/flow")
async def chat_flow(req: ChatRequest):
    """Flow-based chatbot API."""
    if req.user_id not in flow_sessions:
        flow_sessions[req.user_id] = str(uuid.uuid4())

    configurable = {"thread_id": flow_sessions[req.user_id]}
    print("Flow Config:", configurable)

    messages = [HumanMessage(content=req.user_message)]
    response = flow_graph.invoke({"messages": messages}, config={"configurable": configurable})

    print("Flow Response:", response)

    replies = [m.content for m in response["messages"] if isinstance(m, AIMessage)]
    bot_response = replies[-1] if replies else "No response."

    return {"bot_messages": bot_response, "thread_id": flow_sessions[req.user_id]}
