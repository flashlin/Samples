param(
   [string]$action,
   [string]$arg0
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

$env:k8s_exe="C:\Users\flash.lin71\scoop\apps\kubectl\1.24.3\bin\kubectl.exe"
#$env:KUBECONFIG="C:\Users\flash.lin71\.kube\uat.config"
$env:KUBECONFIG="d:\demo\k8s-stg.yaml"

$stateFile = "d:/demo/k8s-state.json" 
$state = [PSCustomObject]@{
   kubeconfig = "D:\demo\k8s-stg.yaml"
   namespace = 'b2c'
   pods = @()
}
$state = GetJsonFile $stateFile $state

function SaveState {
   SetJsonFile $stateFile $state
}

function InvokeK8s {
   param(
      [string]$command
   )
   #InvokeCmd "$($env:k8s_exe) get pods -n b2c" | SplitTableString
   $cmd = "$($env:k8s_exe) $command"
   InvokeCmd $cmd
}


function GetAllPods {
   $state.pods = InvokeK8s "get pods -n $($state.namespace)" | SplitTableString 
   $idx = 0
   $state.pods | ForEach-Object {
      [PSCustomObject]@{
         Id = $idx
         Name = "$($_.NAME)"
         Ready = $_.READY
         Status = $_.STATUS
         Restarts = $_.RESTARTS
      }
      $idx += 1
   }
}

if( "" -eq $action ) {
   Write-Host "reset      : reset k8s script tool state"
   Write-Host "f <name>   : search contain name pattern"
   Write-Host "l [pod id] : logs"
   return
}

if( "reset" -eq $action ) {
   Remove-Item $stateFile
   Write-Host "clean state"
   return
}

if( "f" -eq $action ) {
   $pattern = $arg0
   $myFilter = {
      $_.Name.ToLower() -match $pattern
  }
  $result = GetAllPods | Where-Object -FilterScript $myFilter 
  if( $result.Length -eq 0 ) {
    Write-Host "Not found"
    return
  }
  $state.pods = $result
  SaveState
  $result | Format-Table
  return
}

if( "l" -eq $action ) {
   $id = $arg0
   $myFilter = {
      $_.Id -eq $id
   }
   $pod = $state.pods | Where-Object -FilterScript $myFilter
   InvokeK8s "logs $($pod.Name) -n $($state.namespace)"
   return
}
