function Info {
   param(
      [string]$msg
   )
   Write-Host $msg -ForegroundColor Green
}

function ShowError {
   param(
      [string]$msg
   )
   Write-Host $msg -ForegroundColor Red
}

Import-Module $env:psm1Home/matchUtils.psm1 -Force -Global
Import-Module $env:psm1Home/sqlite.psm1 -Force -Global

function ReadFile {
   param(
      [String]$file
   )
   return Get-Content $file -Encoding UTF8 -Raw
}

function _WriteHostColorFragments {
   param(
      $fragments
   )
   foreach ($fragment in $fragments) {
      if ( $fragment.IsMatch ) {
         Write-Host $fragment.Text -NoNewline -ForegroundColor Green
      }
      else {
         Write-Host $fragment.Text -NoNewline
      }
   }
   Write-Host ""
}

function WriteHostColorText {
   param(
      [string]$text,
      [string[]]$searchPatterns
   )
   $fragments = MatchText $text $searchPatterns
   _WriteHostColorFragments $fragments
}

function CopyToClipboard {
   param(
      [string]$text
   )
   Add-Type -AssemblyName System.Windows.Forms
   [System.Windows.Forms.Clipboard]::SetText($text)
}

function PromptList {
   param(
      [string[]]$items,
      [bool]$descending = $True
   )
   if ( $descending ) {
      $selected = $items | Sort-Object { - $_.Length } -Descending | fzf | Out-String
   }
   else {
      $selected = $items | Sort-Object { - $_.Length } -Descending | fzf | Out-String
   }
   if ( "" -eq $selected ) {
      return ""
   }
   $selected = $selected.TrimEnd()
   return $selected
}

function DumpObject {
   param(
      [object]$obj
   )
   return $obj | ConvertTo-Json
}

function AddObjectProperty {
   param(
      [Parameter(ValueFromPipeline = $true)]
      [object]$obj,
      [string]$name,
      [object]$value
   )
   $obj | Add-Member -MemberType NoteProperty `
      -Name $name -Value $value
}

function Download {
   param (
      [string]$url,
      [string]$targetFile
   )
   Info "Download $url ..."
   Invoke-WebRequest -Uri $url -OutFile $targetFile
}

function Unzip {
   param(
      [string]$zipFile,
      [string]$targetPath
   )
   # 檢查目錄是否存在，如果不存在就建立目錄
   # if (-not (Test-Path -Path $tatgetPath)) {
   #    New-Item -ItemType Directory -Path $targetPath | Out-Null
   # }
   Info "Unzip $zipFile to $targetPath"
   Expand-Archive -Path $zipFile -DestinationPath $targetPath -Force
}

function IsDirectoryExists {
   param(
      [string]$dir
   )
   if (Test-Path -Path $dir) {
      return $True
   }
   return $False
}

function CreateDirectory {
   param(
      [string]$targetPath
   )
   # 檢查目錄是否存在，如果不存在就建立目錄
   if (-not (Test-Path -Path $targetPath)) {
      New-Item -ItemType Directory -Path $targetPath | Out-Null
   }
}

function GetMachinEnvironmentValue {
   param(
      [string]$name
   )
   $value = [Environment]::GetEnvironmentVariable($name, "Machine")
   return $value
}

# choco upgrade chocolatey
function InstallChocolatey {
   $chocoInstallDir = GetMachinEnvironmentValue "ChocolateyInstall"
   if ([string]::IsNullOrWhiteSpace($chocoInstallDir)) {
      Write-Host "Chocolatey is not installed."
      Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
   }
}

function IsChocoPackageExists {
   param(
      $packageName
   )
   #$packageName = "fzf"
   $installedPackages = & choco list
   $packageFound = $installedPackages | Select-String -Pattern $packageName
   return $packageFound
}

function GetJsonFile {
   param (
      [string]$jsonFilePath,
      [object]$defaultValue = [PSCustomObject]@{}
   )
   if (Test-Path $jsonFilePath) {
      $json = Get-Content $jsonFilePath
   }
   else {
      $json = $defaultValue | ConvertTo-Json
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

function DumpProperties {
   param(
      $obj
   )
   $obj | Get-Member -MemberType Property | Format-Table
}


function WriteHostColorByAllMatches {
   param(
      [string]$text,
      [Microsoft.PowerShell.Commands.MatchInfo]$allMatches
   )
   $currentIndex = 0
   foreach ($match in $allMatches.Matches) {
      #DumpProperties $match
      # Write-Host "mi = $($match.Index) '$text' '$($match.Value)'"
      # 輸出未匹配的部分
      if ($match.Index -gt $currentIndex) {
         $unmatchedText = $text.Substring($currentIndex, $match.Index - $currentIndex)
         Write-Host $unmatchedText -NoNewline
      }
      Write-Host $($match.Value) -NoNewline -ForegroundColor Green
      $currentIndex = $match.Index + $match.Length
   }
   # 輸出最後一個未匹配的部分
   if ($currentIndex -lt $text.Length) {
      $unmatchedText = $text.Substring($currentIndex)
      Write-Host $unmatchedText -NoNewline
   }
}

function WriteHostColor {
   param(
      [string]$text,
      [string]$pattern
   )
   if ( '' -eq $pattern ) {
      Write-Host $text -NoNewline
      return
   }
   if ( $null -eq $pattern ) {
      Write-Host $text -NoNewline
      return
   }
   $allMatches = Select-String -InputObject $text -Pattern $pattern -AllMatches
   if ( $null -eq $allMatches ) {
      Write-Host $text -NoNewline
      return
   }
   #DumpProperties $allMatches
   WriteHostColorByAllMatches $text $allMatches
}

function GetFixedText {
   param(
      [string]$text,
      [int]$maxLen
   )
   if ( $text.Length -gt $maxLen ) {
      $sub = $text.Substring(0, $maxLen)
      return $sub
   }
   if ( $text.Length -lt $maxLen ) {
      $spaces = $maxLen - $text.Length
      $sub = $text + "".PadLeft($spaces)
      return $sub
   }
   return $text
}

function WriteFixedText {
   param(
      [string]$text,
      [int]$maxLen
   )
   $sub = GetFixedText $text $maxLen
   Write-Host $sub -NoNewline
}

function WriteHastable {
   param(
      $hash
   )
   foreach ($key in $hash.Keys) {
      $value = $hash[$key]
      Write-Output "$key : $value"
   }
}


Import-Module $env:psm1Home/SplitTableString.psm1 -Force -Global


function WriteTable {
   [CmdletBinding()]
   param(
      [Parameter(ValueFromPipeline = $true)]
      [object]$obj
   )
   begin {
      # 取得最大的長度
      $maxLengthDict = @{} 
      $first = $true
      $list = @()
   }
   process {
      $list += $obj
      if ( $first -eq $true ) {
         $propertyNames = $obj | Get-Member -MemberType Properties | Select-Object -ExpandProperty Name
         foreach ($name in $propertyNames) {
            $val = $obj.$name
            $maxLengthDict[$name] = $val.Length
         }
         $first = $false
      }

      $propertyNames = $propertyNames | Where-Object { !$_.StartsWith("__") }

      foreach ($name in $propertyNames) {
         $val = $obj.$name
         if ( $val.Length -gt $maxLengthDict.$name ) {
            $maxLengthDict[$name] = $val.Length
         }
      }
   }
   end {
      foreach ($name in $propertyNames) {
         $maxLengthDict[$name] += 1
         $maxLen = $maxLengthDict[$name]
         WriteFixedText $name $maxLen
      }
      Write-Host ""
      foreach ($name in $propertyNames) {
         $maxLen = $maxLengthDict[$name]
         $delimit = "-" * $name.Length
         WriteFixedText $delimit $maxLen
      }
      Write-Host ""
      foreach ($item in $list) {
         foreach ($name in $propertyNames) {
            $maxLen = $maxLengthDict[$name]
            $val = $item.$name
            $val = GetFixedText $val $maxLen
            $patternName = "__" + $name
            if ($item | Get-Member -MemberType Properties -Name $patternName ) {
               $pattern = $item.$patternName
               WriteHostColor $val $pattern
            }
            else {
               # WriteFixedText $val $maxLen
               Write-Host $val -NoNewline
            }
         }
         Write-Host ""
      }
   }
}

#  Import-Module $env:psm1Home/docker.psm1 -Force -Global
Export-ModuleMember -Function *