param(
   [string]$action,
   [string]$arg1
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

$env:k8s_exe="C:\Users\flash.lin71\scoop\apps\kubectl\1.24.3\bin\kubectl.exe"
#$env:KUBECONFIG="C:\Users\flash.lin71\.kube\uat.config"
$env:KUBECONFIG="d:\demo\k8s-stg.yaml"
InvokeCmd "$($env:k8s_exe) get pods -n b2c" | SplitTableString