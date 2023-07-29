param(
    [parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$searchPatterns
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

function JumpToFirstDir {
    param(
        [string[]]$dirs,
        [string]$pattern
    )
    if( $dirs.Length -eq 1 ) {
        $dir = $dirs[0]
        WriteHostColor "$dir" $pattern
        Write-Host ""
        Set-Location -Path $dir
    }
}

function DisplayDirs {
    param(
        [string[]]$dirs,
        [string]$pattern
    )
    $index = 0
    $dirs | ForEach-Object {
        $name = $_
        #Write-Host "$($index): $($name)"
        Write-Host "$($index): " -NoNewline
        WriteHostColor "$($name)" $pattern
        Write-Host ""
        $index += 1
    }
}

if( 0 -eq $searchPatterns.Length ) {
    $jj = Get-Content -Path "D:\Demo\jj.txt"
    $searchPatterns = $jj[0] -split "\t"
    $folderNames = $jj[1..($jj.Length - 1)]
    DisplayDirs $folderNames $searchPattern
    return
}

# $folderPattern = $searchPattern
# $regexPattern = "[\[\]\^\$\.\!\=]"
# if (-Not ($searchPattern -match $regexPattern)) {
#     $folderPattern = "^.*$searchPattern.*$"
# }


$notNamePatterns = @(
    "^C:",
    "\\node_modules",
    "\\.git\\",
    "\\.vscode\\",
    "\\.idea\\",
    "\\.vscode-insiders\\",
    "\\.history\\",
    "\\.cache",
    "\\packages",
    "\\bin",
    "\\pkgs\\",
    "\\site-packages",
    "\\dist"
)

function IsValidFolder {
    param(
        [string]$name
    )
    foreach ($namePattern in $notNamePatterns) {
        if ($name -match $namePattern) {
            return $False
        }
    }
    return $True
}

function FindDirectory {
    param(
        [string[]]$names,
        [string]$pattern
    )
    $names = $names | Where-Object { $_ -match $pattern } 
    $names = $names | Where-Object { IsValidFolder $_ }
    return $names
}

function FirstFindDirectory {
    param (
        [string]$pattern
    )
    $result = & es -name-color green /i /ad -regex "$pattern"
    if ($result) {
        $folderNames = $result -split [Environment]::NewLine
        return $folderNames
    }
    return @()
}

function ProcessAllMatchs {
   param(
      [string]$text,
      [Microsoft.PowerShell.Commands.MatchInfo]$allMatches
   )
   $result = @()
   $currentIndex = 0
   foreach ($match in $allMatches.Matches) {
      # 輸出未匹配的部分
      if ($match.Index -gt $currentIndex) {
         $unmatchedText = $text.Substring($currentIndex, $match.Index - $currentIndex)
         $result += @{
            IsMatch = $False
            Text = $unmatchedText
         }
      }
      $result += @{
         IsMatch = $True
         Text = $match.Value
      }
      $currentIndex = $match.Index + $match.Length
   }
   # 輸出最後一個未匹配的部分
   if ($currentIndex -lt $text.Length) {
      $unmatchedText = $text.Substring($currentIndex)
      $result += @{
         IsMatch = $False
         Text = $unmatchedText
      }
   }
   return $result
}


function MatchTextPattern {
    param(
        [string]$text,
        [string]$pattern
    )
    $allMatches = Select-String -InputObject $text -Pattern $pattern -AllMatches
    if( $null -eq $allMatches ) {
        return @(
            @{ 
                IsMatch = $False
                Text = $text 
            }
        )
    }
    return ProcessAllMatchs $text $allMatches
}

function MatchTextFragment {
    param(
        $matchFragments,
        [string]$pattern
    )
    $result = @()
    for($i=0; $i -lt $matchFragments.Length; $i++) {
        $matchFragment = $matchFragments[$i]
        if( $matchFragment.IsMatch ) {
            $result += $matchFragment
            continue
        }
        $subResult = MatchText $matchFragment.Text $pattern
        $result += $subResult
    }
    return $result
}

function MatchText {
    param(
        [string]$text,
        [string[]]$patterns
    )
    $matchFragments = @()
    $first = $true
    foreach ($pattern in $patterns) {
        if( $first ) {
            $matchFragments = MatchTextPattern $text $pattern
            $first = $False
        } else {
            $matchFragments = MatchTextFragment $matchFragments $pattern
        }
    }
    return $matchFragments
}

function WriteHostColorByFragments {
    param(
        $fragments
    )
    foreach($fragment in $fragments) {
        if( $fragment.IsMatch ) {
            Write-Host $fragment.Text -NoNewline -ForegroundColor Green
        } else {
            Write-Host $fragment.Text -NoNewline
        }
    }
    Write-Host ""
}

function WriteColorPath {
    param(
        [string]$path,
        [string[]]$searchPatterns
    )
    $fragments = MatchText $path $searchPatterns
    WriteHostColorByFragments $fragments
}

$paths = FirstFindDirectory $searchPatterns[0]
$paths = $paths | Where-Object { IsValidFolder $_ } 
for ($i = 1; $i -lt $searchPatterns.Length; $i++) {
    $pattern = $searchPatterns[$i]
    $paths = FindDirectory $paths $pattern
}
$paths = $paths | Sort-Object -Property Length, Name -Descending


$searchPatternsText = $searchPatterns -join "\t"
$result = @( $searchPatternsText )
$result += $paths
$result | Set-Content -Path "D:\demo\jj.txt"

if( $paths -is [string] ) {
    WriteColorPath $paths $searchPatterns
    Set-Location -Path $paths
    return
}

$selectedPath = PromptList $paths
if( "" -eq $selectedPath ) {
    return
}
WriteHostColor "$selectedPath" $searchPatterns
Write-Host ""
Set-Location -Path $selectedPath
CopyToClipboard $selectedPath


