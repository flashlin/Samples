
安裝 k3s
```
sudo apt-get update
sudo apt-get install systemd
```

```
sudo vi /etc/wsl.conf
```

修改 `C:\Windows\System32\lxss\tools\initrd.img`
```
[wsl2]
kernel=C:\Windows\System32\lxss\tools\initrd.img
init=/lib/systemd/systemd
```


```
curl -sfL https://get.k3s.io | sh -
sudo systemctl status k3s
```

用管理員身分安裝
```
choco install minikube
minikube start --driver=hyperv 
minikube status
```

推送 image 到 minikube
```
eval $(minikube docker-env)
docker tag my-image:latest minikube/my-image:latest
docker push minikube/my-image:latest
```

eval 將 shell 的環境變量設置為指向 minikube 內部的 Docker 映像庫
docker tag 將 my-image 映像標記為 minikube/my-image

```
kubectl run my-pod --image=minikube/my-image:latest --port=8080
```
這個命令會在 minikube 內運行一個名為 my-pod 的 Pod，使用剛才推送的映像。該映像使用端口 8080。

