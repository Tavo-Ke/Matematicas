"""main.py
Consola interactiva para el sistema de Herramientas Matemáticas con LangGraph.
"""

import sys
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from math_agent import math_agent
from tools import MATH_TOOLS

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BANNER = r"""
================================================================================
   📐 LANGGRAPH MATHEMATICS TOOLS & AGENT SYSTEM
   Herramientas Científicas Autónomas: Álgebra | Cálculo | Matrices | Límites
   Framework: LangGraph + LangChain Tools + SymPy Engine
================================================================================
"""


def print_menu():
    print("\n🛠️  HERRAMIENTAS MATEMÁTICAS REGISTRADAS EN LANGGRAPH:")
    for i, t in enumerate(MATH_TOOLS, 1):
        print(f"  [{i}] {t.name:<26} -> {t.description.split('.')[0]}")
    print("\n💡 Ejemplos de uso directo:")
    print("   • 'Calcula la derivada de x^3 - 6x^2 + 9x'")
    print("   • 'Resuelve la integral de x * sin(x) entre 0 y pi'")
    print("   • 'Calcula el determinante de la matriz [[1, 2], [3, 4]]'")
    print("   • 'Resuelve la ecuacion 2x^2 - 4x - 6 = 0'")
    print("   • 'Calcula el limite de sin(3x)/x cuando x tiende a 0'")
    print("   • 'Factoriza la expresion x^2 - 9'")
    print("   • Escribe 'exit' para salir.\n")


def run_cli():
    print(BANNER)
    print_menu()

    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": f"math_session_{session_id}"}}

    while True:
        try:
            print("-" * 75)
            query = input("🧮 Consulta matemática > ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "salir", "quit", "q"):
                print("\n👋 ¡Hasta luego! Sesión finalizada.")
                break

            print(f"\n⚡ [LangGraph Flow iniciado] -> Procesando con StateGraph...")

            inputs = {"messages": [HumanMessage(content=query)]}
            
            for event in math_agent.stream(inputs, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    print(f"\n🔹 [Nodo: '{node_name}']")
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"   ⚙️  Invocando Herramienta: {tc['name']}")
                                    print(f"      Parámetros: {tc['args']}")
                            else:
                                print(f"   💬 Respuesta:\n{msg.content}")
                        elif isinstance(msg, ToolMessage):
                            print(f"   📊 Resultado de la herramienta:\n{msg.content}")

        except KeyboardInterrupt:
            print("\n\n👋 Operación cancelada por el usuario. Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error durante la ejecución del grafo: {e}")


if __name__ == "__main__":
    run_cli()
