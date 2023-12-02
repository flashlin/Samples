Question: How to install OpenSSH Client for Windows?
Answer:
Run `$PSVersionTable.PSVersion`, 確認您的主要版本至少為 5，且您的次要版本至少為 1
執行以下檢查程式, 以確保您的版本是有符合條件, 假如有符合條件, 結果應該顯示 True
```
(New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
```

請以 管理員 身分執行 安裝 OpenSSH
```
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'
```

安裝之後, 日後可以透過 ssh 連線至 OpenSSH 伺服器
```
ssh domain\username@servername
```