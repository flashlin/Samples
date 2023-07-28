param(
    [string]$action
)
Import-Module "$($env:psm1HOME)/Common.psm1" -Force

if( "" -eq $action ) {
    Write-Host "llama 2 qa demo"
    Write-Host "i : install dependency packages"
    Write-Host "c : create vector stores db"
    Write-Host "r : run"
    return
}

if( "i" -eq $action ) {
    InvokeCmd "pip install -r ./llama2-requirements.txt"
    return
}

if( "c" -eq $action ){
    InvokeCmd "python ./llama2-db.py"
    return
}

if( "r" -eq $action ){
    InvokeCmd "chainlit run .\llama2-langchain.py"
    # InvokeCmd "chainlit run .\llama2-langchain.py -w"  watch
    return
}
