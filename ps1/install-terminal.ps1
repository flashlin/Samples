Install-Module PSReadLine -RequiredVersion 2.1.0 -Force
Set-PSReadLineOption -PredictionSource History
Test-Path $profile
