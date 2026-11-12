@echo off
REM ============================================================
REM  Arrete la stack whisper-npu : serveurs 9200/9201 puis moteur 9199
REM ============================================================
for %%P in (9200 9201 9199) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /r /c:":%%P .*LISTENING"') do (
        echo Arret du PID %%A sur le port %%P
        taskkill /PID %%A /F >nul 2>&1
    )
)
echo Stack whisper-npu arretee.
