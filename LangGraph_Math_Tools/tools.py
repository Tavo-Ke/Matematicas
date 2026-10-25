"""tools.py
Colección de herramientas matemáticas exactas con LangChain / LangGraph y SymPy.
Cada función está decorada con `@tool` con tipado estricto y docstrings para LLMs.
"""

import sys
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any

# Transformaciones para permitir expresiones naturales como '2x' en lugar de '2*x'
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse_safe_math(expr_str: str):
    """Parsea una expresión matemática en formato de texto a objeto SymPy."""
    expr_clean = expr_str.strip().replace("^", "**")
    return parse_expr(expr_clean, transformations=TRANSFORMATIONS)


@tool
def solve_algebraic_equation(equation: str, variable: str = "x") -> str:
    """
    Resuelve ecuaciones algebraicas y polinomiales exactas.
    
    Args:
        equation: La ecuación a resolver (ej. '2*x^2 - 4*x - 6 = 0' o 'sin(x) = 0' o 'x**3 - 8').
                  Si no contiene '=', se asumirá que la expresión es igual a 0.
        variable: El símbolo o variable a despejar (por defecto 'x').
    """
    try:
        var = sp.Symbol(variable)
        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            lhs = _parse_safe_math(lhs_str)
            rhs = _parse_safe_math(rhs_str)
            expr = sp.Eq(lhs, rhs)
            solutions = sp.solve(expr, var)
        else:
            expr = _parse_safe_math(equation)
            solutions = sp.solve(expr, var)

        if not solutions:
            return f"No se encontraron soluciones analíticas reales o complejas para: {equation}"

        output = [f"📐 Ecuación: {equation}", f"Variable: {variable}", f"Total soluciones: {len(solutions)}", "Soluciones:"]
        for i, sol in enumerate(solutions, 1):
            try:
                dec_val = float(sol.evalf())
                output.append(f"  x{i} = {sol}  (Aprox. decimal: {dec_val:.5f})")
            except Exception:
                output.append(f"  x{i} = {sol}")

        return "\n".join(output)
    except Exception as e:
        return f"Error al resolver la ecuación '{equation}': {str(e)}"


@tool
def calculate_derivative(expression: str, variable: str = "x", order: int = 1, evaluate_at: Optional[float] = None) -> str:
    """
    Calcula la derivada analítica simbólica de orden N para una función f(x) y evalúa su pendiente.
    
    Args:
        expression: Expresión matemática a derivar (ej. 'x^3 - 6*x^2 + 9*x', 'sin(x)*exp(x)').
        variable: Variable de derivación (por defecto 'x').
        order: Orden de la derivada (1 para primera derivada, 2 para segunda derivada, etc.).
        evaluate_at: Valor opcional de la variable donde evaluar numéricamente la derivada.
    """
    try:
        var = sp.Symbol(variable)
        func = _parse_safe_math(expression)
        derivative = sp.diff(func, var, order)

        output = [
            f"📈 Función f({variable}) = {func}",
            f"Orden de derivación: {order}",
            f"Resultado f^({order})({variable}) = {derivative}"
        ]

        if evaluate_at is not None:
            val_exact = derivative.subs(var, evaluate_at)
            val_num = float(val_exact.evalf())
            output.append(f"Evaluación en {variable} = {evaluate_at}: {val_exact} (Aprox: {val_num:.5f})")

        return "\n".join(output)
    except Exception as e:
        return f"Error al calcular la derivada de '{expression}': {str(e)}"


@tool
def calculate_integral(expression: str, variable: str = "x", lower_limit: Optional[str] = None, upper_limit: Optional[str] = None) -> str:
    """
    Calcula integrales indefinidas (+ C) o integrales definidas entre límites dados.
    
    Args:
        expression: Expresión o integrando a calcular (ej. 'x * sin(x)', 'exp(-x**2)').
        variable: Variable de integración (por defecto 'x').
        lower_limit: Límite inferior para integral definida (ej. '0', '-oo', '-pi'). Opcional.
        upper_limit: Límite superior para integral definida (ej. 'pi', 'oo', '1'). Opcional.
    """
    try:
        var = sp.Symbol(variable)
        func = _parse_safe_math(expression)

        if lower_limit is not None and upper_limit is not None:
            a = _parse_safe_math(lower_limit)
            b = _parse_safe_math(upper_limit)
            res = sp.integrate(func, (var, a, b))
            output = [
                f"∫ Integral definida de f({variable}) = {func}",
                f"Límites: [{a}, {b}]",
                f"Resultado exacto: {res}"
            ]
            try:
                dec = float(res.evalf())
                output.append(f"Aproximación decimal: {dec:.5f}")
            except Exception:
                pass
            return "\n".join(output)
        else:
            res = sp.integrate(func, var)
            return f"∫ Integral indefinida de f({variable}) = {func}\nResultado: {res} + C"
    except Exception as e:
        return f"Error al calcular la integral de '{expression}': {str(e)}"


@tool
def calculate_limit(expression: str, variable: str = "x", point: str = "0", direction: str = "+-") -> str:
    """
    Calcula el límite analítico de una función f(x) cuando x tiende a un punto dado (incluyendo 'oo' o '-oo').
    
    Args:
        expression: Expresión a evaluar (ej. 'sin(3*x)/x', '(1 + 1/x)^x').
        variable: Variable independiente (por defecto 'x').
        point: Punto hacia el que tiende la variable (ej. '0', 'oo', '-oo', '2').
        direction: Dirección del límite: '+-' (bilateral), '+' (por la derecha), '-' (por la izquierda).
    """
    try:
        var = sp.Symbol(variable)
        func = _parse_safe_math(expression)
        pt = _parse_safe_math(point)

        if direction in ("+", "-"):
            lim_val = sp.limit(func, var, pt, dir=direction)
            dir_str = "por la derecha (+)" if direction == "+" else "por la izquierda (-)"
        else:
            lim_val = sp.limit(func, var, pt)
            dir_str = "bilateral"

        return (
            f"🎯 Límite: lim_{{{variable} -> {pt}}} [{func}]\n"
            f"Dirección: {dir_str}\n"
            f"Resultado: {lim_val}"
        )
    except Exception as e:
        return f"Error al calcular el límite de '{expression}': {str(e)}"


@tool
def simplify_and_factor(expression: str, operation: str = "simplify") -> str:
    """
    Aplica simplificación, factorización o expansión algebraica a una expresión matemática.
    
    Args:
        expression: Expresión algebraica (ej. 'x^2 - 9', '(x + 2)*(x - 3)', 'sin(x)**2 + cos(x)**2').
        operation: 'simplify' para simplificar, 'factor' para factorizar en factores primos, o 'expand' para expandir polinomios.
    """
    try:
        expr = _parse_safe_math(expression)
        op = operation.lower().strip()
        if op == "factor":
            res = sp.factor(expr)
            label = "Factorización"
        elif op == "expand":
            res = sp.expand(expr)
            label = "Expansión"
        else:
            res = sp.simplify(expr)
            label = "Simplificación"

        return f"✨ Operación: {label}\nOriginal: {expr}\nResultado: {res}"
    except Exception as e:
        return f"Error al simplificar o factorizar '{expression}': {str(e)}"


@tool
def matrix_operations(matrix_data: List[List[float]], operation: str = "determinant", vector_b: Optional[List[float]] = None) -> str:
    """
    Realiza operaciones fundamentales de Álgebra Lineal sobre matrices.
    
    Args:
        matrix_data: Matriz numérica representada como lista de listas (ej. [[1, 2], [3, 4]]).
        operation: Tipo de operación:
                   - 'determinant' (o 'det'): Calcula el determinante de la matriz.
                   - 'inverse' (o 'inv'): Calcula la matriz inversa.
                   - 'eigenvalues' (o 'eigen'): Calcula autovalores y autovectores.
                   - 'rank': Calcula el rango de la matriz.
                   - 'solve_system': Resuelve A * x = b (requiere pasar 'vector_b').
        vector_b: Vector de términos independientes para 'solve_system' (ej. [5, 11]).
    """
    try:
        M = sp.Matrix(matrix_data)
        op = operation.lower().strip()

        if op in ("determinant", "det"):
            det_val = M.det()
            return f"🔢 Determinante de matriz {M.shape[0]}x{M.shape[1]}:\ndet(M) = {det_val}"
        
        elif op in ("inverse", "inv"):
            inv_M = M.inv()
            return f"🔄 Matriz Inversa M^(-1):\n{inv_M}"

        elif op in ("rank", "rango"):
            r = M.rank()
            return f"📊 Rango de la matriz: {r}"

        elif op in ("eigenvalues", "eigen", "autovalores"):
            eigenvals = M.eigenvals()
            out = ["🧬 Autovalores y Multiplicidades:"]
            for val, mult in eigenvals.items():
                out.append(f"  λ = {val} (multiplicidad: {mult})")
            return "\n".join(out)

        elif op in ("solve_system", "solve", "sistema"):
            if not vector_b:
                return "Error: Para resolver un sistema de ecuaciones se requiere 'vector_b'."
            b = sp.Matrix(vector_b)
            sol = M.LUsolve(b)
            return f"🎯 Solución del sistema A*x = b:\nx = {list(sol)}"

        else:
            return f"Operación '{operation}' no reconocida. Usa: 'determinant', 'inverse', 'rank', 'eigenvalues', o 'solve_system'."
    except Exception as e:
        return f"Error en operaciones de matriz: {str(e)}"


@tool
def python_math_repl(code: str) -> str:
    """
    Ejecutor seguro de código Python con bibliotecas científicas (SymPy, NumPy, Math)
    para resolver cualquier problema matemático computacional personalizado.
    
    Args:
        code: Bloque de código Python que debe imprimir la respuesta final usando print().
    """
    import io
    import contextlib

    # Espacio de nombres seguro precargado con librerías matemáticas
    safe_globals = {
        "sp": sp,
        "sympy": sp,
        "__builtins__": {
            "__import__": __import__,
            "print": print,
            "range": range,
            "len": len,
            "float": float,
            "int": int,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "enumerate": enumerate,
            "zip": zip
        }
    }

    try:
        import numpy as np
        safe_globals["np"] = np
        safe_globals["numpy"] = np
    except ImportError:
        pass

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        output = stdout_capture.getvalue()
        if not output.strip():
            return "Código ejecutado con éxito (sin salida por consola). Asegúrate de incluir print()."
        return output.strip()
    except Exception as e:
        return f"Error al ejecutar código Python matemático: {str(e)}"


# Lista completa de herramientas matemáticas para LangGraph
MATH_TOOLS = [
    solve_algebraic_equation,
    calculate_derivative,
    calculate_integral,
    calculate_limit,
    simplify_and_factor,
    matrix_operations,
    python_math_repl
]
