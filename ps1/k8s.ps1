param(
   [string]$action,
   [string]$arg0,
   [string]$arg1
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

$env:k8s_exe="C:\Users\flash.lin71\scoop\apps\kubectl\1.24.3\bin\kubectl.exe"
$env:k8s_exe="kubectl"
#$env:KUBECONFIG="C:\Users\flash.lin71\.kube\uat.config"
$env:KUBECONFIG="d:\demo\k8s-stg.yaml"

$kubeConfigDict = @{
   "stg" = "k8s-stg.yaml"
   "gstg" = "host-staging-gke.yaml"
   "uat" = "host-uat-gke.yaml"
}


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

function ShowState {
   Write-Host "kubeconfig: $($state.kubeconfig)"
   Write-Host "namespace: $($state.namespace)"
}

function InvokeK8s {
   param(
      [string]$command
   )
   #InvokeCmd "$($env:k8s_exe) get pods -n b2c" | SplitTableString
   $env:KUBECONFIG = $state.kubeconfig
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

function GetPodNameFromSelected {
   param(
      [int]$id
   )
   $myFilter = {
      $_.Id -eq $id
   }
   $pod = $state.pods | Where-Object -FilterScript $myFilter
   return $pod.Name.Trim()
}

if( "" -eq $action ) {
   Write-Host "reset               : reset k8s script tool state"
   Write-Host "f <name>            : search contain name pattern"
   Write-Host "l [pod id]          : logs"
   Write-Host "use <env>           : use stg/uat environment"
   Write-Host "ns [namespace]      : list namespaces / switch to namespace(b2c)"
   Write-Host "cp [pod id] [source]: copy pod's source to d:\demo\k8s "
   Write-Host ""
   ShowState
   return
}

if( "use" -eq $action ) {
   $staging = $arg0
   $state.kubeconfig = "D:\demo\" + $kubeConfigDict.($staging)
   SaveState
   Write-Host "use $staging $($state.kubeconfig)"
   ShowState
   return
}

if( "ns" -eq $action ) {
   $namespace = $arg0
   if( "" -eq $arg0 ) {
      $namespaceList = InvokeK8s "get namespaces"
      if( $namespaceList.Length -eq 0 ) {
         return
      }
      $selectedNamespace = PromptList $namespaceList
      Write-Host $selectedNamespace
      return
   }
   # $namespace = "product-platform"
   $state.namespace = $namespace
   SaveState
   Write-Host "switch to $($state.namespace)"
   ShowState
   return
}

if( "cp" -eq $action ) {
   $id = $arg0
   if( $null -eq $id ) {
      Write-Host "cp <pod-id> /usr/xxx"
      return
   }

   $podName = GetPodNameFromSelected $id
   $source = $arg1
   $targetFilename = Split-Path -Leaf $source
   InvokeK8s "cp $($state.namespace)/$($podName):$source ./$($targetFilename)"
   # InvokeK8s "cp -n $($state.namespace) $($podName):$source d:\demo\k8s"
   return
}

if( "ls" -eq $action ) {
   $id = $arg0
   if( $null -eq $id ) {
      Write-Host "ls <pod-id>"
      return
   }
   $podName = GetPodNameFromSelected $id
   $dir = $arg1
   InvokeK8s "exec -n $($state.namespace) $($podName) -- ls $($dir) |sort"
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
  ShowState
  return
}

if( "l" -eq $action ) {
   $id = $arg0
   $pattern = $arg1
   $myFilter = {
      $_.Id -eq $id
   }
   $pod = $state.pods | Where-Object -FilterScript $myFilter
   InvokeK8s "logs $($pod.Name) -n $($state.namespace)" | ForEach-Object {
      $allMatches = MatchText $_ $pattern
      if( $null -ne $allMatches ) {
         WriteHostColorByAllMatches $_ $allMatches
         Write-Host ""
      }
      # WriteHostColor $_ $pattern
   }
   return
}
