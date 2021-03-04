@echo off

@REM Clean the build folder i.e. remove old files/directories
call make clean

@REM Remove old versions of .rst files
rm ".\source\face*" ".\source\modules.rst"

@REM Re-create .rst files
sphinx-apidoc -o ./source ../face_recognizer

@REM Re-create the final html file
call make html

@REM The following code block asks the user if he/she wants to open the index.html generated above
setlocal
:PROMPT
SET /P AREYOUSURE=Open the html file (Y/[N])?
IF /I "%AREYOUSURE%" NEQ "Y" GOTO END

@REM Note regarding the following command line: the first "" after the 'start' command is mandatory. see: https://ss64.com/nt/start.html 
start "" ".\build\html\index.html"

:END
endlocal