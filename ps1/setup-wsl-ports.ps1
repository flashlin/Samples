$DISTRO="Ubuntu-22.04"

$wsl_ip = (& wsl -d $DISTRO -- bash -c "ifconfig eth0 | grep 'inet '").trim().split() | Where-Object {$_}
$regex = [regex] "\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"

$ip_array = $regex.Matches($wsl_ip) | ForEach-Object{ $_.value }
$wsl_ip = $ip_array[0]

Write-Host "WSL Machine IP: ""$wsl_ip"""


#### ------------ Delete PortProxy rules ------------ ####
netsh int portproxy reset all

#### ------------ Rule: SSH - Port 22 ------------ ####
$export_ports = @( 
   (8880, 8880), 
   (3306, 3306),
   (8001, 8001),
   (6333, 6333)
)

function InvokeCmdAsAdmin {
   param(
      [string]$cmd,
      [string]$arguments
   )
   Start-Process -FilePath $cmd -ArgumentList $arguments -Verb RunAs
}

$export_ports | ForEach-Object {
   $wslPort = $_[0]
   $listenPort = $_[1]
   Write-Host "listenPort $listenPort -> $wslPort"
   $arguments = "interface portproxy add v4tov4 listenport=$listenPort listenaddress=0.0.0.0 connectport=$wslPort connectaddress=$wsl_ip"
   #netsh interface portproxy add v4tov4 listenport=$listenPort listenaddress=0.0.0.0 connectport=$wslPort connectaddress=$wsl_ip
   InvokeCmdAsAdmin "netsh" $arguments
}

& netsh interface portproxy show all