@echo off

REM Set the output filename and location
set OUTPUT_FILE=%~dp0\submission.tar.gz

REM Change to the directory where the files are located
cd %~dp0

REM Archive the files using tar and gzip
tar -czf "%OUTPUT_FILE%" main.py agent.py components/* lux/*

echo "Files archived successfully to %OUTPUT_FILE%."