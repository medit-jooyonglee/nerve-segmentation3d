@REM python
@REM import test
@REM test.test_dicom_reader

@REM https://stackoverflow.com/questions/4580101/python-add-pythonpath-during-command-line-module-run
@REM python -c "import sys; sys.path.append('/your/script/path'); import yourscript; yourscript.yourfunction()"

@ECHO OFF
setlocal
@REM https://2ry53.tistory.com/entry/Bat-%ED%8C%8C%EC%9D%BC-%EC%8B%A4%ED%96%89%EB%90%9C-bat%ED%8C%8C%EC%9D%BC-%EC%9C%84%EC%B9%98-%EC%95%8C%EA%B8%B0-%EC%8B%A4%ED%96%89%EA%B2%BD%EB%A1%9C%EB%A1%9C-%EC%9D%B4%EB%8F%99%ED%95%98%EA%B8%B0
ECHO %~dp0
@REM ECHO 현재 디렉토리 - %cd%


@REM set PYTHONPATH=%cd%\..;
set PYTHONPATH=%~dp0\..;
ECHO %PYTHONPATH%
python %~dp0\..\dataset\prepare_dataset.py
endlocal