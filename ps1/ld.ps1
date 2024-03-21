param (
    #[Parameter(Mandatory, Position = 0)]
    [string]$search_text = ""
)

function MatchString {
    param(
       [string]$text,
       [string]$searchPattern
    )
    $rc = $text -match $searchPattern
    if ( $rc -eq $true) {
        return $true
    }
    return $false
 }

if ( "" -ne $search_text ) {
    Get-ChildItem -Directory | Where-Object { MatchString $_.Name $search_text } | `
        Select-Object -Property Name | Format-Wide -Column 4
    return
}

if ( "-t" -ne $search_text ) {
    Write-Host "Directory 依照建立時間排序, 從舊到新"
    Get-ChildItem -Directory | Sort-Object CreationTime | `
        Select-Object -Property Name | Format-Wide -Column 4
    return
}

Get-ChildItem -Directory -Filter $search_text | `
    Select-Object -Property Name | Format-Wide -Column 4
