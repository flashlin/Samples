function GetJsonFile {
   param (
      [string]$jsonFilePath
   )
   if (Test-Path $jsonFilePath) {
      $json = Get-Content $jsonFilePath
   }
   else {
      $json = "{}"
   }
   $obj = $json | ConvertFrom-Json
   return $obj
}

function SetJsonFile {
   param (
      $jsonFilePath,
      $obj
   )
   $json = $obj | ConvertTo-Json
   $json | Out-File -FilePath $jsonFilePath -Force
}

function InvokeCmd {
   param(
      [string]$cmd
   )
   Write-Host $cmd -ForegroundColor Green
   Invoke-Expression $cmd
}

#  Import-Module $env:psm1Home/docker.psm1 -Force -Global
Export-ModuleMember -Function *