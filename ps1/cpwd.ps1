$workPath = (Get-Location).Path
Set-Clipboard -Value $workPath
Write-Host "'$workPath' to Clipboard."