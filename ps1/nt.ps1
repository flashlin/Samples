$connectIp="10.12.33.80"
netsh interface portproxy add v4tov4 `
    listenport=11434 `
    listenaddress=127.0.0.1 `
    connectport=11434 `
    connectaddress=$connectIp