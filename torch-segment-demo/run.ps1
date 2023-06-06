param(
    [string]$action,
    [string]$arg0
)
Import-Module "$($env:psm1HOME)/common.psm1" -Force

InvokeCmd "hug -f web_segmentation.py -ho 0.0.0.0 -p 8081"
