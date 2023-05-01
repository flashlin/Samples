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

function WriteHostColor {
   param(
      [string]$text,
      [string]$pattern
   )
   if ( $null -eq $pattern ) {
      Write-Host $text -NoNewline
      return
   }
   $allMatches = Select-String -InputObject $text -Pattern $pattern -AllMatches
   $currentIndex = 0
   foreach ($match in $allMatches.Matches) {
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
      if ( $first -eq $true ) {
         $propertyNames = $obj | Get-Member -MemberType Properties | Select-Object -ExpandProperty Name
         foreach ($name in $propertyNames) {
            $val = $obj.$name
            $maxLengthDict[$name] = $val.Length
         }
         $first = $false
      }

      # $propertyNames = $propertyNames | ForEach-Object {
      #    if ($_.Name.StartsWith("__")) {
      #       continue
      #    }
      # }

      foreach ($name in $propertyNames) {
         $val = $obj.$name
         if ( $val.Length -gt $maxLengthDict.$name ) {
            $maxLengthDict[$name] = $val.Length
         }
      }
      $list += $obj
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
      foreach ($item in $list) {
         foreach ($name in $propertyNames) {
            $maxLen = $maxLengthDict[$name]
            $val = $item.$name
            # $val = GetFixedText $val $maxLen
            # $patternName = "__" + $item.$name
            # $pattern = $item.$patternName
            # WriteHostColor $val $pattern
            WriteFixedText $val $maxLen
         }
         Write-Host ""
      }
   }
}

#  Import-Module $env:psm1Home/docker.psm1 -Force -Global
Export-ModuleMember -Function *