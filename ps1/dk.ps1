param(
   [string]$action,
   [string]$arg1
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
Import-Module "$($env:psm1HOME)/docker.psm1" -Force

Write-Host "Docker Tool v1.0 by flash"
Write-Host ""

$env:docker_exe = "docker"

$dkOptionsFilePath = "d:\demo\dk.json"
$dkOptions = GetJsonFile $dkOptionsFilePath
if ( $null -eq $dkOptions.docker ) {
   $dkOptions | Add-Member -MemberType NoteProperty -Name "docker" -Value "docker"
}
SetJsonFile $dkOptionsFilePath $dkOptions

if ( "use" -eq $action ) {
   $ans = Read-Host "use podman? (y/n)"
   if ( $ans -eq "y" ) {
      $env:docker_exe = "podman"
      $dkOptions.docker = "podman"
      SetJsonFile $dkOptionsFilePath $dkOptions
   }
   Write-Host "use $($env:docker_exe)"
   return
}

$env:docker_exe = $dkOptions.docker
Write-Host "use $($env:docker_exe)"

#$keyword = "flas"
# | ForEach-Object {
#    $item = $_
#    $item | Add-Member -MemberType NoteProperty -Name "__Repo" -Value "flas"
#    $item
# } | WriteTable


if( "rmi" -eq $action) {
   $name = $arg1
   if( "" -eq $name ) {
      $name = "<none>"
   }
   Write-Host "clean all $name images..."
   QueryDockerImages $name | ForEach-Object {
      Write-Host "remove $($_.Repo) $($_.Tag) ..."
      InvokeDocker "rmi $($_.ID) -f"
   }
   return
}

if( "i" -eq $action ){
   QueryDockerImages "" 
   return
}