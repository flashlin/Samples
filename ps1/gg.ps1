param (
   [Parameter(Position = 0)]
   [string]$choiceIndex = 0
)
Import-Module "$($env:psm1HOME)/Common.psm1" -Force
Import-Module "$($env:psm1HOME)/ConsoleCore.psm1" -Force

#$findFolder = New-Object ConsoleCore.FindFolder
#$folder = $findFolder.Choice("D:\Demo", $choiceIndex)


$searcher = New-Object ConsoleCore.FolderSearcher
$folder = $searcher.GetSearchResult($choiceIndex)

Write-Host $folder -ForegroundColor Green
Set-Location $folder
$searcher.AddFolder($folder)
cpwd
