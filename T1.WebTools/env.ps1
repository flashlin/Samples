$env:docker_exe = "docker"
$ans = Read-Host "use podman? (y/n)"
if ( $ans -eq "y" ) {
   $env:docker_exe = "podman"
}
Write-Host "use $($env:docker_exe)"