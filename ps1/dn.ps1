param(
    [string]$url
)
$file = Split-Path $url -leaf
Invoke-WebRequest -Uri $url -OutFile "./$file"