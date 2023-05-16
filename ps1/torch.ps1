Import-Module $env:psm1Home\common.psm1 -Force

$cmd = "nvcc -V"
InvokeCmd $cmd

$cmd = "python $($env:psm1HOME)/../pym1/torch-env.py"
InvokeCmd $cmd