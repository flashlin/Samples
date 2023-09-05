Import-Module "$($env:psm1HOME)/common.psm1" -Force
$ErrorActionPreference = "Stop"

function FileIsNotExists {
    param(
        [string]$file
    )
    if (-not (Test-Path -Path $file -PathType Leaf)) {
        return $True
    }
    return $False
}

function IsDirectoryNotExists {
    param(
        [string]$dir
    )
    if (-not (Test-Path -Path $dir)) {
        return $True
    }
    return $False
}

function RemoveSystemEnvironment {
    param (
        [string]$pathPattern 
    )
    $envName = "PATH"
    $envValue = [Environment]::GetEnvironmentVariable($envName, "Machine")
    
    $paths = $envValue -split ";"
    
    $removePaths = $paths | Where-Object { $_ -match $pathPattern }
    foreach ($path in $removePaths) {
        $paths = $paths | Where-Object { $_ -ne $path }
    }

    $newEnvValue = $paths -join ';'
    [Environment]::SetEnvironmentVariable($envName, $newEnvValue, "Machine")
    $env:Path = $newEnvValue
}

function AddSystemEnvironment {
    param(
        [string]$newPath
    )
    $envName = "PATH"
    $envValue = [Environment]::GetEnvironmentVariable($envName, "Machine")
    if ($envValue -notlike "*$($newPath)*") {
        $newEnvValue = $envValue + ";$($newPath)"
        [Environment]::SetEnvironmentVariable($envName, $newEnvValue, "Machine")
        $env:Path = $newEnvValue
        Info "Add '$($newPath)' to System Environment Path"
    }
}

function InvokeCmdAsAdmin {
    param(
        [string]$cmd
    )
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$cmd`"" -Verb RunAs
}

$targetPath = "C:\Program Files\nvim-win64"
if ( IsDirectoryNotExists $targetPath ) {
    Info "Downloading nvim-win64"
    $targetFile = "./nvim-win64.zip"
    if (-not (Test-Path -Path $targetFile)) {
        $zipUrl = "https://github.com/neovim/neovim/releases/download/v0.9.1/nvim-win64.zip"
        Download $zipUrl $targetFile
    }
    Unzip $targetFile "C:\Program Files"

    Info "Remove Neovim from System Environment Path"
    RemoveSystemEnvironment "nvim"
    Info "Add '$($targetPath)\bin' to System Environment Path"
    AddSystemEnvironment "$($targetPath)\bin"
}


#InvokeCmd $cmd
$RoamingPath = $env:APPDATA
$AppDataPath = $RoamingPath + "\.."
$AppLocalPath = $AppDataPath + "\Local"
$NeoVimConfigPath = $AppLocalPath + "\nvim"

Info "copy init.vim to $NeoVimConfigPath"
Copy-Item -Path ./neovim-data/* -Destination $NeoVimConfigPath -Recurse -Container -Force


$NeoVimAutoloadPath = "$NeoVimConfigPath\autoload"
if ( IsDirectoryNotExists $NeoVimAutoloadPath ) {
    Info "Install PlugInstall Manager..."
    CreateDirectory $NeoVimAutoloadPath
    $uri = "https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim"
    Download $uri "$NeoVimAutoloadPath\plug.vim"
}

InstallChocolatey

if ( -Not (IsChocoPackageExists "fzf") ) {
    choco install fzf
}

#choco install ag
#winget install "The Silver Searcher"