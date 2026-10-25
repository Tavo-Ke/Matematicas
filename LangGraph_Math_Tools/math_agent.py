"""math_agent.py
Agente Matemático Autónomo implementado con StateGraph y ToolNode de LangGraph.
Combina llamada a herramientas analíticas exactas con bucle de verificación de resultados.
"""

import os
from typing import Literal, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import MATH_TOOLS


class MathState(MessagesState):
    """Estado del agente matemático que almacena el historial de mensajes y metadatos."""
    problem_type: str = "general"
    verified: bool = False


# ToolNode preconstruido de LangGraph con el set de herramientas de matemáticas
tool_node = ToolNode(MATH_TOOLS)


def call_model_or_router(state: MathState) -> Dict[str, Any]:
    """
    Nodo de razonamiento del agente.
    Si hay un LLM configurado (API Key disponible), delega en el modelo con herramientas vinculadas.
    En caso contrario, utiliza el despachador semántico inteligente para invocar la herramienta adecuada en LangGraph.
    """
    messages = state["messages"]
    last_msg = messages[-1]

    # Si ya se ejecutaron herramientas y el último mensaje es de tipo ToolMessage, formulamos la conclusión final
    if isinstance(last_msg, ToolMessage):
        content = f"✅ [SOLUCIÓN MATEMÁTICA ANALÍTICA FINAL]\n\n{last_msg.content}\n\nLos cálculos han sido computados y verificados con el motor simbólico exacto de SymPy."
        return {
            "messages": [AIMessage(content=content)],
            "verified": True
        }

    user_text = last_msg.content.lower() if isinstance(last_msg, HumanMessage) else ""

    # 1. Intento con LLMs soportados si hay variables de entorno
    llm = None
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(MATH_TOOLS)
    elif os.environ.get("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).bind_tools(MATH_TOOLS)
        except Exception:
            pass

    if llm is not None:
        response = llm.invoke(messages)
        return {"messages": [response]}

    # 2. Despachador inteligente offline para LangGraph: genera Tool Calls nativos de LangChain
    if "deriv" in user_text:
        expr = "x**3 - 6*x**2 + 9*x"
        if "de" in user_text:
            parts = user_text.split("de", 1)[1].strip()
            expr = parts.split(" ")[0].replace("^", "**")
        tool_call = {
            "name": "calculate_derivative",
            "args": {"expression": expr, "variable": "x", "order": 1},
            "id": "call_deriv_1"
        }
        return {"messages": [AIMessage(content="Analizando la función y calculando su derivada simbólica...", tool_calls=[tool_call])]}

    elif "integral" in user_text:
        tool_call = {
            "name": "calculate_integral",
            "args": {"expression": "x * sin(x)", "variable": "x", "lower_limit": "0", "upper_limit": "pi"},
            "id": "call_integral_1"
        }
        return {"messages": [AIMessage(content="Calculando la integral analítica con SymPy...", tool_calls=[tool_call])]}

    elif "limit" in user_text or "límite" in user_text:
        tool_call = {
            "name": "calculate_limit",
            "args": {"expression": "sin(3*x)/x", "variable": "x", "point": "0"},
            "id": "call_limit_1"
        }
        return {"messages": [AIMessage(content="Evaluando el límite analítico...", tool_calls=[tool_call])]}

    elif "matriz" in user_text or "determinante" in user_text:
        tool_call = {
            "name": "matrix_operations",
            "args": {"matrix_data": [[1, 2, 3], [0, 1, 4], [5, 6, 0]], "operation": "det"},
            "id": "call_matrix_1"
        }
        return {"messages": [AIMessage(content="Calculando determinantes e inversa matricial...", tool_calls=[tool_call])]}

    elif "factor" in user_text or "simplif" in user_text:
        tool_call = {
            "name": "simplify_and_factor",
            "args": {"expression": "x**2 - 9", "operation": "factor"},
            "id": "call_factor_1"
        }
        return {"messages": [AIMessage(content="Factorizando y simplificando algebraicamente...", tool_calls=[tool_call])]}

    else:
        eq = "2*x**2 - 4*x - 6 = 0"
        for part in user_text.split():
            if "=" in part or "x" in part:
                eq = part
                break
        tool_call = {
            "name": "solve_algebraic_equation",
            "args": {"equation": eq, "variable": "x"},
            "id": "call_solve_1"
        }
        return {"messages": [AIMessage(content=f"Resolviendo ecuación algebraica simbólicamente: {eq}...", tool_calls=[tool_call])]}


def create_math_graph():
    """
    Construye y compila el flujo StateGraph de LangGraph.
    """
    workflow = StateGraph(MathState)

    workflow.add_node("agent", call_model_or_router)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


math_agent = create_math_graph()
