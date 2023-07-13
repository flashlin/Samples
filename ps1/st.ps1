$answer = Read-Host "set shutdown time: (min)"
$sec = [int]$answer
$sec = $sec * 60
$cmd = "shutdown /s /f /t $sec"
Write-Host $cmd
Invoke-Expression $cmd
