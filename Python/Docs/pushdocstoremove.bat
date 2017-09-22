@echo off

SET ROOT="%~dp0\.."

REM build the docs
cd %ROOT%\Docs
make clean
make html
xcopy /s /e /y build "%TEMP%\build\"
cd %ROOT%

REM switch branches and pull the data we want
git checkout gh-pages
cd ..
del /Q %ROOT%\*
for /d %x in (%ROOT%\*) do @rd /S /Q "%x"
cd %ROOT%
copy NUL .nojekyll
xcopy /e /h /y /c "%TEMP%\build\html" .\
git add -A
git commit -m "publishing updated docs..."
git push origin gh-pages

REM switch back
git checkout master