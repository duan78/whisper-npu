@echo off
REM ============================================================
REM  Stack serveurs whisper-npu (depuis ce depot)
REM  0. whisper-server      http://127.0.0.1:9199  moteur persistant NPU
REM                         modele ggml + cache .rai residants, charges 1 fois
REM  1. npu-asr-server      http://127.0.0.1:9200/v1  facade OpenAI sur 9199
REM                         repli whisper-cli si le moteur est down
REM  2. voice-stream-server ws://127.0.0.1:9201       Realtime-like
REM  Prerequis : Lemonade lance - port 9000 - pour le mode agent
REM ============================================================
cd /d "%~dp0"
set ENGINE=C:\Users\Arnaud\dev\whisper.cpp\build-pipe\bin\Release\whisper-server.exe
set MODEL=C:\Users\Arnaud\dev\whisper-npu-bin\models\ggml-large-v3-turbo.bin

if not exist silero_vad.onnx (
    echo [ERREUR] silero_vad.onnx manquant dans %CD%
    echo           voir docs/06-voice-loop.md pour le recuperer
    pause
    exit /b 1
)

netstat -ano | findstr /r /c:":9199 .*LISTENING" >nul
if not errorlevel 1 goto already9199

echo Demarrage du moteur persistant whisper-server - 9199 - chargement ~10 s...
start "Whisper-Engine (whisper-npu)" /min cmd /c "%ENGINE% -m %MODEL% -l fr --host 127.0.0.1 --port 9199 -bs 1 -t 16"
set /a tries=0

:wait9199
ping -n 2 127.0.0.1 >nul
netstat -ano | findstr /r /c:":9199 .*LISTENING" >nul
if not errorlevel 1 goto ready9199
set /a tries+=1
if %tries% lss 30 goto wait9199
echo [ATTENTION] moteur 9199 pas pret - npu-asr-server utilisera le repli whisper-cli
goto servers

:ready9199
echo Moteur persistant pret - 9199.
goto servers

:already9199
echo Moteur persistant deja lance - 9199.

:servers
netstat -ano | findstr /r /c:":9200 .*LISTENING" >nul
if not errorlevel 1 goto skip9200
echo Demarrage npu-asr-server - port 9200...
start "NPU-ASR-Server (whisper-npu)" cmd /k python "%~dp0npu-asr-server.py"
:skip9200

ping -n 3 127.0.0.1 >nul
netstat -ano | findstr /r /c:":9201 .*LISTENING" >nul
if not errorlevel 1 goto skip9201
echo Demarrage voice-stream-server - port 9201...
start "Voice-Stream-Server (whisper-npu)" cmd /k python "%~dp0voice-stream-server.py"
:skip9201

ping -n 3 127.0.0.1 >nul
curl -s http://127.0.0.1:9200/health
echo.
echo Stack prete : POST http://127.0.0.1:9200/v1/audio/transcriptions
