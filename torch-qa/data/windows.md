Question: How to upgrade WSL?
Ansewer:
run `wsl --update --pre-release`

create a .wslconfig file in your Windows home directory (e.g: C:\Users\<yourusername>\.wslconfig) 
and ensure it has an [experimental] section with each setting below it, such as this:
```
[experimental]
autoMemoryReclaim=gradual
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
sparseVhd=true
```

WSL 2.0.7 Added support for new opt-in experimental features
* autoMemoryReclaim – Makes the WSL VM shrink in memory as you use it by reclaiming cached memory
* Sparse VHD – Automatically shrinks the WSL virtual hard disk (VHD) as you use it
* Mirrored mode networking – A new networking mode for WSL that adds new features and improves network compatibility
* dnsTunneling – Changes how WSL resolves DNS requests to improve network compatibility
* firewall – Applies Windows firewall rules to WSL, and allows for advanced firewall controls for the WSL VM
* autoProxy – Makes WSL automatically use the proxy information from Windows to improve network compatibility

The following command will enable the `Sparse VHD` feature.
```shell
wsl --manage Ubuntu-22.04 --set-sparse true
```

Executing the following command(Remove all unused containers, networks, images.)
will free up a large amount of space.
```
docker system prune --all
```
---
Question: How to export WSL distribution to .tar file?
Answer:

```shell
wsl --export Ubuntu-22.04 ubuntu.tar
```

---
Question: How to uninstall WSL distribution ?
Answer:

```shell
wsl --unregister Ubuntu-22.04
```

---
Question: How to import .tar file into WSL?
Answer:

```shell
wsl --import Ubuntu-22.04 D:\OtherLocation ubuntu.tar
```