param($script,$log)
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="cmd.exe /c powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script > $log 2>&1"} | select ReturnValue
