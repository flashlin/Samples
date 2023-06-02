param(
   [string]$action,
   [string]$arg1
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

$env:k8s_exe="C:\Users\flash.lin71\scoop\apps\kubectl\1.24.3\bin\kubectl.exe"
#$env:KUBECONFIG="C:\Users\flash.lin71\.kube\uat.config"
$env:KUBECONFIG="d:\demo\k8s-stg.yaml"

$stateFile = "d:/demo/k8s-state.json" 
$state = [PSCustomObject]@{
   kubeconfig = "D:\demo\k8s-stg.yaml"
   pods = @()
}
$state = GetJsonFile $stateFile $state


function InvokeK8s {
   param(
      [string]$command
   )
   #InvokeCmd "$($env:k8s_exe) get pods -n b2c" | SplitTableString
   InvokeCmd "$($env:k8s_exe) $command"
}


$state.pods = InvokeK8s "get pods -n b2c" | SplitTableString 
SetJsonFile $stateFile $state
$idx = 0
$state.pods | ForEach-Object {
   [PSCustomObject]@{
      Name = "$idx - $($_.NAME)"
      Ready = $_.READY
      Status = $_.STATUS
      Restarts = $_.RESTARTS
   }
   $idx += 1
} | Format-Table



