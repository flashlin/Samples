Set-ExecutionPolicy RemoteSigned
[Environment]::SetEnvironmentVariable("psm1HOME", "D:\vdisk\github\Samples\ps1\psm1", [System.EnvironmentVariableTarget]::User)
[Environment]::SetEnvironmentVariable("path", $env:Path + ";D:\VDisk\Github\Samples\ps1", [System.EnvironmentVariableTarget]::Machine)
