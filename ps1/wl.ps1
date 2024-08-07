param(
    [string]$action
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

Write-Host "wsl script"

$result = InvokeCmd "wsl -l -v" | SplitTableString
$result | ForEach-Object {
    Write-Host $_
}

function AskWslNames {
    param (
    )
    $names = $result | ForEach-Object {
        $_.NAME
    }
    $name = PromptList $names
    return $name
}

if ( 's' -eq $action ) {
    $names = $result | ForEach-Object {
        $_.NAME
    }
    $name = PromptList $names
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --shutdown $name"
    return
}


if ( 'b' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl -d $name"
    return
}


if ( 'r' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --shutdown $name"
    InvokeCmd "wsl -d $name"
    return
}

if ( 'bb' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl -d $name -- bash --noprofile"
    return
}

if( 'cfg' -eq $action ) {
    # wsl --update --pre-release
    # $userName = $env:USERNAME
    $wslConfig = "$env:USERPROFILE/.wslconfig"
    if (Test-Path $wslConfig) {
        Write-Host "$wslConfig is Exists."
    } else {
        Copy-Item -Path .\.wslconfig -Destination $wslConfig -Force
        Write-Host "$wslConfig done."
    }
    return
}

if ( 'rm' -eq $action ) {
    $name = AskWslNames
    if ( "" -eq $name ) {
        return
    }
    InvokeCmd "wsl --unregister $name" 
    return
}

if ( "i" -eq $action ) {
    InvokeCmd "wsl --install -d Ubuntu-22.04"
    return
}

if( "net" -eq $action ) {
    $result = Invoke-Expression "wsl -d Ubuntu-22.04 hostname -I"
    $connectIp = $result.Split(' ')[0]
    Write-Host $connectIp
    $ports = @( (2222,20), (8888,8888) )
    foreach ($connectPort in $ports) {
        $port = $tuple[0]
        $connectPort = $tuple[1]
        Write-Host "listenPort $port"
        netsh interface portproxy add v4tov4 `
            listenport=$port `
            listenaddress=0.0.0.0 `
            connectport=$connectPort `
            connectaddress=$connectIp
    }
    $ruleName = "WSL"
    $existingRule = Get-NetFirewallRule -DisplayName $ruleName
    if ($null -eq $existingRule) {
        #Remove-NetFirewallRule -DisplayName $ruleName
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -LocalPort $ports -Protocol TCP -Action Allow
    }
    netsh interface portproxy show all
    return
}

Write-Host ""
Write-Host "i  :install Ubuntu-22.04"
Write-Host "s  :stop"
Write-Host "b  :run and enter"
Write-Host "r  :restart and enter"
Write-Host "bb :run and enter directly"
Write-Host "rm :delete"
Write-Host "net:add WSL listen ports"
Write-Host "cfg:init .wslconfig file"