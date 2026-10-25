"""test_tools.py
Pruebas automatizadas de las herramientas matemáticas y el flujo de LangGraph.
"""

import unittest
from tools import (
    solve_algebraic_equation,
    calculate_derivative,
    calculate_integral,
    calculate_limit,
    simplify_and_factor,
    matrix_operations,
    python_math_repl
)
from math_agent import math_agent
from langchain_core.messages import HumanMessage


class TestMathTools(unittest.TestCase):

    def test_solve_algebraic_equation(self):
        res = solve_algebraic_equation.invoke({"equation": "2*x^2 - 4*x - 6 = 0", "variable": "x"})
        self.assertIn("-1", res)
        self.assertIn("3", res)

    def test_calculate_derivative(self):
        res = calculate_derivative.invoke({"expression": "x^3 - 6*x^2 + 9*x", "variable": "x", "order": 1, "evaluate_at": 1.0})
        self.assertIn("3*x**2 - 12*x + 9", res)
        self.assertIn("0.00000", res)

    def test_calculate_integral(self):
        res = calculate_integral.invoke({"expression": "x", "variable": "x", "lower_limit": "0", "upper_limit": "2"})
        self.assertIn("2", res)

    def test_calculate_limit(self):
        res = calculate_limit.invoke({"expression": "sin(3*x)/x", "variable": "x", "point": "0"})
        self.assertIn("3", res)

    def test_simplify_and_factor(self):
        res = simplify_and_factor.invoke({"expression": "x^2 - 9", "operation": "factor"})
        self.assertIn("(x - 3)*(x + 3)", res)

    def test_matrix_operations(self):
        res = matrix_operations.invoke({"matrix_data": [[1, 2], [3, 4]], "operation": "det"})
        self.assertIn("-2", res)

    def test_python_math_repl(self):
        code = "import sympy as sp\nx = sp.Symbol('x')\nprint(sp.expand((x+1)**3))"
        res = python_math_repl.invoke({"code": code})
        self.assertIn("x**3 + 3*x**2 + 3*x + 1", res)

    def test_langgraph_flow(self):
        config = {"configurable": {"thread_id": "test_thread_1"}}
        inputs = {"messages": [HumanMessage(content="Calcula la derivada de x^3 - 6*x^2")]}
        final_state = math_agent.invoke(inputs, config=config)
        
        self.assertGreater(len(final_state["messages"]), 1)
        last_message = final_state["messages"][-1]
        self.assertIn("SOLUCIÓN MATEMÁTICA", last_message.content)


if __name__ == "__main__":
    unittest.main()
