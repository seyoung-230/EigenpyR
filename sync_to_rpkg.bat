@echo off
setlocal enabledelayedexpansion

REM === 기본 경로(프로젝트 루트 기준) ===
set ROOT=%~dp0
set CORE_INC=%ROOT%core\include
set CORE_SRC=%ROOT%core\src
set PKG_ROOT=%ROOT%bindings\r\EigenRBinding
set PKG_INC=%PKG_ROOT%\inst\include
set PKG_SRC=%PKG_ROOT%\src

echo [EigenPyR] Core -> R package sync 시작
echo   ROOT      : %ROOT%
echo   CORE_INC  : %CORE_INC%
echo   CORE_SRC  : %CORE_SRC%
echo   PKG_INC   : %PKG_INC%
echo   PKG_SRC   : %PKG_SRC%

REM 폴더 생성
if not exist "%PKG_INC%" mkdir "%PKG_INC%"
if not exist "%PKG_SRC%" mkdir "%PKG_SRC%"

REM include 동기화: /E 하위폴더 포함, /XO 최신파일 유지
robocopy "%CORE_INC%" "%PKG_INC%" /E /XO
REM src 동기화: module.cpp는 제외
robocopy "%CORE_SRC%" "%PKG_SRC%" /E /XO /XF module.cpp

REM robocopy 종료코드 0~7은 성공 범주
set RC=%ERRORLEVEL%
if %RC% LEQ 7 (
  echo [EigenPyR] 복사 완료
) else (
  echo [EigenPyR] 복사 중 오류 발생 (robocopy code %RC%)
)

echo 이제 bindings\r\EigenRBinding\src\module.cpp, R\api.R만 수정하면 됩니다.
pause
