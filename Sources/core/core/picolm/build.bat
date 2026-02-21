@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo Compiling...
cl /O2 /W3 /Fe:picolm.exe picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
) else (
    echo BUILD SUCCESS
)
