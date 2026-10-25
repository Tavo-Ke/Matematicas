"""web_app.py
Servidor Web FastAPI para el Sistema de Herramientas Matemáticas con LangGraph + Google Gemini + Artifacts UI.
Integra renderizado de gráficas interactivas, notación KaTeX y visualización del StateGraph en vivo.
"""

import os
import sys
import re
import json
import uuid
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from math_agent import math_agent, DEFAULT_GEMINI_API_KEY, DEFAULT_GEMINI_MODEL
from tools import MATH_TOOLS

app = FastAPI(title="LangGraph Math Artifacts Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    query: str
    api_key: Optional[str] = DEFAULT_GEMINI_API_KEY
    model: Optional[str] = DEFAULT_GEMINI_MODEL
    thread_id: Optional[str] = "web_session_1"


@app.get("/api/config")
async def get_config():
    tools_info = [
        {
            "name": t.name,
            "description": t.description.strip().split("\n")[0]
        }
        for t in MATH_TOOLS
    ]
    return {
        "default_api_key": DEFAULT_GEMINI_API_KEY,
        "default_model": DEFAULT_GEMINI_MODEL,
        "available_models": [
            "gemini-3.8-flash",
            "gemini-4-argon",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ],
        "tools": tools_info
    }


@app.post("/api/solve")
async def solve_math_problem(req: SolveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="La consulta matemática no puede estar vacía.")

    api_key = (req.api_key or DEFAULT_GEMINI_API_KEY).strip()
    model_name = (req.model or DEFAULT_GEMINI_MODEL).strip()
    thread_id = req.thread_id or f"session_{uuid.uuid4().hex[:6]}"

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=req.query)],
        "api_key": api_key,
        "model_name": model_name
    }

    steps_trace = []
    invoked_tool_name = None
    invoked_tool_args = {}
    raw_tool_output = ""
    final_answer = ""

    try:
        for event in math_agent.stream(inputs, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            tc = msg.tool_calls[0]
                            invoked_tool_name = tc["name"]
                            invoked_tool_args = tc["args"]
                            steps_trace.append({
                                "node": node_name,
                                "type": "tool_call",
                                "tool": invoked_tool_name,
                                "args": invoked_tool_args,
                                "summary": f"Agente Gemini ({model_name}) seleccionó @{invoked_tool_name}"
                            })
                        else:
                            final_answer = msg.content
                            steps_trace.append({
                                "node": node_name,
                                "type": "synthesis",
                                "summary": "Síntesis analítica y generación de Artifacts KaTeX/JS completada"
                            })
                    elif isinstance(msg, ToolMessage):
                        raw_tool_output = msg.content
                        steps_trace.append({
                            "node": node_name,
                            "type": "tool_execution",
                            "tool": invoked_tool_name or "sympy_tool",
                            "summary": "ToolNode ejecutó cálculo simbólico exacto con SymPy"
                        })

        plot_artifacts = []
        for match in re.finditer(r"<mathplot>([\s\S]*?)</mathplot>", final_answer):
            try:
                plot_artifacts.append(json.loads(match.group(1).strip()))
            except Exception:
                pass

        if not plot_artifacts and raw_tool_output:
            for match in re.finditer(r"<mathplot>([\s\S]*?)</mathplot>", raw_tool_output):
                try:
                    plot_obj = json.loads(match.group(1).strip())
                    plot_artifacts.append(plot_obj)
                    final_answer += f"\n<mathplot>\n{json.dumps(plot_obj)}\n</mathplot>"
                except Exception:
                    pass

        active_tool = invoked_tool_name or "solve_algebraic_equation"
        mermaid_code = f"""flowchart LR
    START([START]) --> Agent1["🤖 Agent Node\\n({model_name})"]
    Agent1 -- "tool_calls:\\n{active_tool}" --> ToolNode["🛠️ ToolNode\\n(@{active_tool})"]
    ToolNode -- "SymPy Exact Result\\n+ LaTeX + Plot Data" --> Agent2["📐 Synthesizer\\n(KaTeX + Artifacts)"]
    Agent2 --> END([END])
    style Agent1 fill:#1e3a8a,stroke:#60a5fa,color:#fff
    style ToolNode fill:#065f46,stroke:#34d399,color:#fff
    style Agent2 fill:#4c1d95,stroke:#a78bfa,color:#fff"""

        return {
            "answer": final_answer,
            "raw_sympy": re.sub(r"<mathplot>[\s\S]*?</mathplot>", "", raw_tool_output).strip(),
            "tool_used": active_tool,
            "tool_args": invoked_tool_args,
            "model_used": model_name,
            "steps": steps_trace,
            "plots": plot_artifacts,
            "mermaid": mermaid_code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ejecución de LangGraph: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    print("🚀 Servidor Web LangGraph Math Artifacts iniciado en: http://localhost:8085")
    uvicorn.run(app, host="0.0.0.0", port=8085)
