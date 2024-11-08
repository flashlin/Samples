param (
   [Parameter(Mandatory, Position = 0)]
   [string]$text,
   [Parameter(Mandatory, Position = 1)]
   [string]$files
)
$workPath = (Get-Location).Path

$excludeFolders = @(
   "/node_modules",
   "/debug",
   "/bin",
   "/obj",
   "/dist",
   "/.git",
   "/.vs",
   "/packages",
   "/.deploy",
   "/cache"
)

function MatchExcludeFolders {
   param(
      [string]$path
   )
   $folder = $path.Replace("\", "/")
   for ($idx = 0; $idx -lt $excludeFolders.Length; $idx++) {
      $pattern = $excludeFolders[$idx]
      $rc = $folder -match $pattern
      if ( $rc -eq $true) {
         return $true
      }
   }
   return $false
}

function GetFolders { 
   param(
      [string]$path
   )
   Get-ChildItem -Path $path -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
      $folder = $_.Fullname
      $m = MatchExcludeFolders $folder
      if ( $m -eq $true ) {
         return
      }
      GetFolders($folder) | ForEach-Object {
         Write-Output $_
      }
      Write-Output $folder
   }
}

$foldersCount = 0
$host.privatedata.ProgressForegroundColor = "white";
$host.privatedata.ProgressBackgroundColor = "darkgreen";

#Get-ChildItem -Path $workPath -Directory -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
#   $folder = $_.Fullname
GetFolders($workPath) | ForEach-Object {
   $folder = $_
   $foldersCount += 1
   Write-Progress -Activity $folder -Status "[$foldersCount]..."
   # $m = MatchExcludeFolders $folder
   # if ( $m -eq $true ) {
   #    return
   # }

   Get-ChildItem "$($folder)" -Filter "$($files)" `
   | Select-String -Pattern $text `
   | Select-Object Path, LineNumber, Line `
   | Format-Table
}

function GetFolderDeepLength {
   param(
      [string]$folder
   )
   $charCount = ($folder.ToCharArray() | Where-Object { $_ -eq '\' } | Measure-Object).Count
   return $charCount
}
