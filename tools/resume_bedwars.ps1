$q = 'C:\Users\benja\qonly-staging'
Remove-Item "$q\runs\autopilot\STOP" -ErrorAction SilentlyContinue
"STOP present after delete: " + (Test-Path "$q\runs\autopilot\STOP")
$cl = 'cmd /c C:\Users\benja\AppData\Local\Programs\Python\Python310\python.exe -B C:\Users\benja\qonly-staging\code-v\pc\autopilot.py --config C:\Users\benja\qonly-staging\runs\autopilot-config.json >> C:\Users\benja\qonly-staging\runs\autopilot\v.out 2>&1'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cl; CurrentDirectory="$q\code-v"}
"autopilot started pid " + $r.ProcessId + " rc " + $r.ReturnValue + " at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
