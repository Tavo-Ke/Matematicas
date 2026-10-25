@echo off
chcp 65001 > nul
title LangGraph Math Artifacts Studio (Gemini + SymPy + KaTeX)
cd /d "%~dp0"
echo ============================================================================
echo   Iniciando Servidor Web LangGraph Math Artifacts en http://localhost:8085
echo ============================================================================
start http://localhost:8085
python web_app.py
pause
