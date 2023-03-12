# $env:Path = "D:\devp\MinGW\bin;" + $env:Path
$env:Path = "D:\Devp\Dev-Cpp\bin;" + $env:Path
gcc -o system-io.wasm -nostartfiles "-Wl,--no-entry" "-Wl,--export-all" "-Wl,--allow-undefined" system-io.c
