:: Open OpenFlux-gui

set "venv=talltower"

@call conda activate %venv%

@if %ERRORLEVEL% == 0 goto Script

@call %HOMEPATH%\Anaconda3\Scripts\activate.bat %HOMEPATH%\Anaconda3

@call activate %venv%

@if %ERRORLEVEL% neq 0 goto CuldeSac

@goto Script

:CuldeSac

@echo "ANACONDA NOT FOUND. Make sure Anaconda is installed (https://www.anaconda.com/)"

@msg * "ANACONDA NOT FOUND. Make sure Anaconda is installed (https://www.anaconda.com/)"

exit /b 1

:Script

@cd %~dp0/../..

@if exist "%~dp0logo.ico" set "logo=%~dp0logo.ico"
@if exist "%~dp0logo.png" set "logo=%~dp0logo.png"
@if exist "%~dp0logo.jpg" set "logo=%~dp0logo.jpg"

@python __gargantua__.py path="Lib/open_flux/setup/readme.yaml" lib="open_flux" welcometxt="OpenFLUX" font="small"

@echo %ERRORLEVEL%

:End

@if %ERRORLEVEL% == 9 call %~dp0OpenFlux-gui.bat"

@if %ERRORLEVEL% neq 9 if %ERRORLEVEL% neq 0 pause

exit /b 0