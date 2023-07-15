Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

InstallChocolatey

Write-Host "checking fzf package..."
if( -Not (IsChocoPackageExists "fzf") ) {
    choco install fzf
}
