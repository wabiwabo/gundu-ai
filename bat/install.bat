@echo off
cd  C:\Users\gundu\Documents\gundu-ai\gundu-ai
py -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
pause