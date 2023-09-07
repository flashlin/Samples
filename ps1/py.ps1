param(
    [string]$script
)

$command = "python.exe $($script)"
for ($i = 0; $i -lt $args.count; $i++ ) {
    $arg = $args[$i]
    $command += " $($arg)"
}

Write-Host $command -ForegroundColor Green
Invoke-Expression $command