#!/bin/bash
# usage: launch.sh <name> <python args...>   -> runs detached on PC in project dir, log to logs\<name>.log
name=$1; shift
args="$*"
PS="\$cl = 'cmd /c cd /d C:\\Users\\benja\\premonition-projectors && set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& venv\\Scripts\\python.exe -B $args >> logs\\$name.log 2>&1'; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\$cl; CurrentDirectory='C:\\Users\\benja\\premonition-projectors'}; 'pid ' + \$r.ProcessId"
$(dirname $0)/ps.sh "$PS"
