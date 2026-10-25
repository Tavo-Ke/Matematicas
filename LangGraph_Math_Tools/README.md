# 📐 LangGraph Math Tools & Agent System

Suite de herramientas matemáticas de alta precisión integradas en **LangGraph** y **LangChain** para la resolución exacta, analítica y simbólica de problemas de matemáticas avanzadas utilizando **SymPy**, **NumPy** y **Python REPL**.

---

## 🚀 Características Principales

1. **Herramientas Nativas `@tool` de LangChain/LangGraph**:
   - `solve_algebraic_equation`: Resolución exacta de ecuaciones algebraicas y polinomiales con raíces analíticas y decimales.
   - `calculate_derivative`: Cálculo simbólico de derivadas de orden $n$, con evaluación de pendientes y puntos críticos.
   - `calculate_integral`: Integrales indefinidas ($+ C$) e integrales definidas con límites numéricos o simbólicos ($\pi, \infty$).
   - `calculate_limit`: Límites analíticos bilaterales y unilaterales ($x \to 0, \infty, -\infty$).
   - `simplify_and_factor`: Simplificación, factorización y expansión de expresiones complejas.
   - `matrix_operations`: Determinantes, matrices inversas, rango, autovalores/autovectores y solución de sistemas $A \cdot x = b$.
   - `python_math_repl`: Entorno seguro de ejecución Python con SymPy y NumPy para cálculos libres.

2. **Arquitectura StateGraph con ToolNode**:
   - Ciclo reactivo `START` $\to$ `agent` $\rightleftarrows$ `tools` $\to$ `END`.
   - Soporte para memoria conversacional mediante `MemorySaver`.
   - Capacidad híbrida: compatible con modelos LLM (OpenAI, Gemini) y con enrutador determinista offline.

---

## 🏗️ Diagrama de Arquitectura (LangGraph)

```mermaid
flowchart TD
    START([START]) --> AgentNode["🤖 Agente / Razonador"]
    AgentNode --> Condition{"¿Requiere Herramienta?"}
    Condition -- Sí (tool_calls) --> ToolNode["🛠️ ToolNode (SymPy Tools)"]
    ToolNode --> AgentNode
    Condition -- No (Respuesta Lista) --> END([END])

    subgraph Herramientas ["Colección de Herramientas LangGraph"]
        T1["solve_algebraic_equation"]
        T2["calculate_derivative"]
        T3["calculate_integral"]
        T4["calculate_limit"]
        T5["simplify_and_factor"]
        T6["matrix_operations"]
        T7["python_math_repl"]
    end

    ToolNode -.-> Herramientas
```

---

## 📦 Instalación y Requisitos

```bash
pip install -r requirements.txt
```

---

## 💻 Ejecución

### 1. Ejecutar la consola interactiva:
```bash
python main.py
```

### 2. Ejecutar los tests unitarios automatizados:
```bash
python test_tools.py
```
