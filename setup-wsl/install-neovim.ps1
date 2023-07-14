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

function Download {
    param (
        [string]$url,
        [string]$targetFile
    )
    Info "Download $url ..."
    Invoke-WebRequest -Uri $url -OutFile $targetFile
}

function CreateDir {
    param(
        [string]$targetPath
    )
    # 檢查目錄是否存在，如果不存在就建立目錄
    if (-not (Test-Path -Path $targetPath)) {
       New-Item -ItemType Directory -Path $targetPath | Out-Null
    }
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

$targetFile = "./nvim-win64.zip"
if (-not (Test-Path -Path $targetFile)) {
    $zipUrl = "https://github.com/neovim/neovim/releases/download/v0.9.1/nvim-win64.zip"
    Download $zipUrl $targetFile
}

$targetPath = "C:\Program Files"
#Unzip $targetFile $targetPath

Info "Remove Neovim from System Environment Path"
RemoveSystemEnvironment "nvim"
Info "Add '$($targetPath)\nvim-win64\bin' to System Environment Path"
AddSystemEnvironment "$($targetPath)\nvim-win64\bin"

#InvokeCmd $cmd
$RoamingPath = $env:APPDATA
$AppDataPath = $RoamingPath+"\.."
$AppLocalPath = $AppDataPath+"\Local"
$NeoVimConfigPath = $AppLocalPath+"\nvim"

Info "copy init.vim to $NeoVimConfigPath"
Copy-Item -Path ./neovim-data/* -Destination $NeoVimConfigPath -Recurse -Container -Force


Info "Install PlugInstall Manager..."
$NeoVimAutoloadPath = "$NeoVimConfigPath\autoload"
CreateDir $NeoVimAutoloadPath
$uri = "https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim"
Download $uri "$NeoVimAutoloadPath\plug.vim"
