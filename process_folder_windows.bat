@echo off
setlocal EnableDelayedExpansion

rem Check if the required argument is provided
if "%~1" == "" (
    echo Usage: %~nx0 folder
    exit /b 1
)

rem Get the folder path
set "folder=%~1"

rem Check if the folder exists
if not exist "%folder%" (
    echo Folder %folder% does not exist.
    exit /b 1
)

rem Loop over audio files in the folder
for %%F in ("%folder%\*.mp3" "%folder%\*.wav" "%folder%\*.ogg" "%folder%\*.flac") do (
    if exist "%%F" (
        echo Processing file: %%F
        python3 diarize.py --audio "%%F" ^
                              --no-stem ^
                              --suppress_numerals ^
                              --whisper-model "openai/whisper-medium.en" ^
                              --batch-size 8 ^
                              --language en ^
                              --device "cuda" ^
                              --out-dir "C:\path\to\output\directory"
    )
)

endlocal
