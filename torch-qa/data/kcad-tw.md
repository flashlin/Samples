# CKAD 模擬考試 50 題：核心概念、多容器 Pod、Pod 設計 - 知乎

隨著 Kubernetes 的發展，如今各大企業對於能在 Kubernetes 上建構應用程式的開發人員的需求也在不斷增長。2018 年 5 月，CNCF 曾推出 CKAD 考試，旨在
**考察工程師是否具備在 Kubernetes 上設計、建構、組態和公開雲原生應用程式的能力**。

K8sMeetup 中國社區在此提供了 50 道練習題，幫助開發者測試自己的技術熟練程度。

**核心概念**
--------
請根據以下概念進行練習：瞭解 Kubernetes API 原語，建立和組態基本 Pod。

Question: 列出叢集中的所有命名空間
Answer:
```
kubectl get namespaces kubectl get ns
```

Question: 列出所有命名空間中的所有 Pod
Answer:
```
kubectl get po --all-namespaces
```


3.列出特定命名空間中的所有 Pod
```
kubectl get po -n <namespace name>
```


4.列出特定命名空間中的所有 Service
```
kubectl get svc -n <namespace name>
```


5.用 json 路徑表示式列出所有顯示名稱和命名空間的 Pod

```
kubectl get pods -o=jsonpath="{.items[*]['metadata.name', 'metadata.namespace']}"
```


6.在默認命名空間中建立一個 Nginx Pod，並驗證 Pod 是否正在運行
```
// creating a pod
kubectl run nginx --image=nginx --restart=Never
// List the pod
kubectl get po
```


7.使用 yaml 檔案建立相同的 Nginx Pod
```
// get the yaml file with --dry-run flag
kubectl run nginx --image=nginx --restart=Never --dry-run -o yaml > nginx-pod.yaml
// cat nginx-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    run: nginx
  name: nginx
spec:
  containers:
  - image: nginx
    name: nginx
    resources: {}
  dnsPolicy: ClusterFirst
  restartPolicy: Never
status: {}
// create a pod 
kubectl create -f nginx-pod.yaml
```


8.輸出剛建立的 Pod 的 yaml 檔案

```
kubectl get po nginx -o yaml
```


9.輸出剛建立的 Pod 的 yaml 檔案，並且其中不包含特定於叢集的資訊

```
kubectl get po nginx -o yaml --export
```


10.獲取剛剛建立的 Pod 的完整詳細資訊

```
kubectl describe pod nginx
```


11.刪除剛建立的 Pod

```
kubectl delete po nginx
kubectl delete -f nginx-pod.yaml
```


12.強制刪除剛建立的 Pod

```
kubectl delete po nginx --grace-period=0 --force
```


13.建立版本為 1.17.4 的 Nginx Pod，並將其暴露在連接埠 80 上

```
kubectl run nginx --image=nginx:1.17.4 --restart=Never --port=80
```


14.將剛建立的容器的鏡像更改為 1.15-alpine，並驗證該鏡像是否已更新

```
kubectl set image pod/nginx nginx=nginx:1.15-alpine
kubectl describe po nginx
// another way it will open vi editor and change the version
kubeclt edit po nginx
kubectl describe po nginx
```


15.對於剛剛更新的 Pod，將鏡像版本改回 1.17.1，並觀察變化

```
kubectl set image pod/nginx nginx=nginx:1.17.1
kubectl describe po nginx
kubectl get po nginx -w # watch it
```


16.在不使用 describe 命令的情況下檢查鏡像版本

```
kubectl get po nginx -o jsonpath='{.spec.containers[].image}{"\n"}'
17.建立 Nginx Pod 並在 Pod 上執行簡單的 shell
// creating a pod
kubectl run nginx --image=nginx --restart=Never
// exec into the pod
kubectl exec -it nginx /bin/sh
```


18.獲取剛剛建立的 Pod 的 IP 地址

```
kubectl get po nginx -o wide
```


19.建立一個 busybox Pod，在建立它時運行命令 ls 並檢查日誌

```
kubectl run busybox --image=busybox --restart=Never -- ls kubectl logs busybox
```


20.如果 Pod 崩潰了，請檢查 Pod 的先前日誌

21.使用命令 sleep 3600 建立一個 busybox Pod

```
kubectl run busybox --image=busybox --restart=Never -- /bin/sh -c "sleep 3600"
```


22.檢查 busybox Pod 中 Nginx Pod 的連接

```
kubectl get po nginx -o wide
// check the connection
kubectl exec -it busybox -- wget -o- <IP Address>
```


23.建立一個能回顯消息“How are you”的 busybox Pod，並手動將其刪除

```
kubectl run busybox --image=nginx --restart=Never -it -- echo "How are you"
kubectl delete po busybox
```


24.建立一個能回顯消息“How are you”的 busybox Pod，並手動將其刪除

```
// notice the --rm flag kubectl run busybox --image=nginx --restart=Never -it --rm -- echo "How are you"
```


25.建立一個 Nginx Pod 並列出具有不同複雜度（verbosity）的 Pod

```
// create a pod
kubectl run nginx --image=nginx --restart=Never --port=80
// List the pod with different verbosity
kubectl get po nginx --v=7
kubectl get po nginx --v=8
kubectl get po nginx --v=9
```


26.使用自訂列 PODNAME 和 PODSTATUS 列出 Nginx Pod

```
kubectl get po -o=custom-columns="POD_NAME:.metadata.name, POD_STATUS:.status.containerStatuses[].state"
```


27.列出所有按名稱排序的 Pod

```
kubectl get pods --sort-by=.metadata.name
```


28.列出所有按建立時間排序的 Pod

```
kubectl get pods--sort-by=.metadata.creationTimestamp
```


多容器 Pod
-------

請根據以下概念進行練習：瞭解多容器 Pod 的設計模式（例：ambassador、adaptor、sidecar）。

29.用“ls; sleep 3600;”“echo Hello World; sleep 3600;”及“echo this is the third container; sleep 3600”三個命令建立一個包含三個 busybox 容器的 Pod，並觀察其狀態

```
// first create single container pod with dry run flag
kubectl run busybox --image=busybox --restart=Never --dry-run -o yaml -- bin/sh -c "sleep 3600; ls" > multi-container.yaml
// edit the pod to following yaml and create it
kubectl create -f multi-container.yaml
kubectl get po busybox
```


30.檢查剛建立的每個容器的日誌

```
kubectl logs busybox -c busybox1
kubectl logs busybox -c busybox2
kubectl logs busybox -c busybox3
```


31.檢查第二個容器 busybox2 的先前日誌（如果有）

```
kubectl logs busybox -c busybox2 --previous
```


32.在上述容器的第三個容器 busybox3 中運行命令 ls

```
kubectl exec busybox -c busybox3 -- ls
```


33.顯示以上容器的 metrics，將其放入 file.log 中並進行驗證

```
kubectl top pod busybox --containers
// putting them into file
kubectl top pod busybox --containers > file.log
cat file.log
```


34.用主容器 busybox 建立一個 Pod，並執行“while true; do echo ‘Hi I am from Main container’ >> /var/log/index.html; sleep 5; done”，並帶有暴露在連接埠 80 上的 Nginx 鏡像的 sidecar 容器。用 emptyDir Volume 將該卷安裝在 /var/log 路徑（用於 busybox）和 /usr/share/nginx/html 路徑（用於nginx容器）。驗證兩個容器都在運行

```
// create an initial yaml file with this
kubectl run multi-cont-pod --image=busbox --restart=Never --dry-run -o yaml > multi-container.yaml
// edit the yml as below and create it
kubectl create -f multi-container.yaml
kubectl get po multi-cont-pod
```


35.進入兩個容器並驗證 main.txt 是否存在，並用 `curl localhost` 從 sidecar 容器中查詢 main.txt

```
// exec into main container
kubectl exec -it  multi-cont-pod -c main-container -- sh
cat /var/log/main.txt
// exec into sidecar container
kubectl exec -it  multi-cont-pod -c sidecar-container -- sh
cat /usr/share/nginx/html/index.html
// install curl and get default page
kubectl exec -it  multi-cont-pod -c sidecar-container -- sh
# apt-get update && apt-get install -y curl
# curl localhost
```


Pod 設計
------

請根據以下概念進行練習：瞭解如何使用 Labels、Selectors 和 Annotations，瞭解部署以及如何執行滾動更新，瞭解部署以及如何執行回滾，瞭解 Jobs 和 CronJobs.

36.獲取帶有標籤資訊的 Pod

```
kubectl get pods --show-labels
```


37.建立 5 個 Nginx Pod，其中兩個標籤為 env = prod，另外三個標籤為 env = dev

```
kubectl run nginx-dev1 --image=nginx --restart=Never --labels=env=dev
kubectl run nginx-dev2 --image=nginx --restart=Never --labels=env=dev
kubectl run nginx-dev3 --image=nginx --restart=Never --labels=env=dev
kubectl run nginx-prod1 --image=nginx --restart=Never --labels=env=prod
kubectl run nginx-prod2 --image=nginx --restart=Never --labels=env=prod
```


38.確認所有 Pod 都使用正確的標籤建立

```
kubeclt get pods --show-labels
```


39.獲得帶有標籤 env = dev 的 Pod

```
kubectl get pods -l env=dev
```


40.獲得帶有標籤 env = dev 的 Pod，並輸出標籤

```
kubectl get pods -l env=dev --show-labels
```


41.獲得帶有標籤 env = prod 的 Pod

```
kubectl get pods -l env=prod
```


42.獲得帶有標籤 env = prod 的 Pod，並輸出標籤

```
kubectl get pods -l env=prod --show-labels
```


43.獲取帶有標籤 env 的 Pod

44.獲得帶有標籤 env = dev 和 env = prod 的 Pod

```
kubectl get pods -l 'env in (dev,prod)'
```


45.獲取帶有標籤 env = dev 和 env = prod 的 Pod 並輸出標籤

```
kubectl get pods -l 'env in (dev,prod)' --show-labels
```


46.將其中一個容器的標籤更改為 env = uat 並列出所有要驗證的容器

```
kubectl label pod/nginx-dev3 env=uat --overwrite 
kubectl get pods --show-labels
```


47.刪除剛才建立的 Pod 標籤，並確認所有標籤均已刪除

```
kubectl label pod nginx-dev{1..3} env-
kubectl label pod nginx-prod{1..2} env-
kubectl get po --show-labels
```


48.為所有 Pod 新增標籤 app = nginx 並驗證

```
kubectl label pod nginx-dev{1..3} app=nginx
kubectl label pod nginx-prod{1..2} app=nginx
kubectl get po --show-labels
```


49.獲取所有帶有標籤的節點（如果使用 minikube，則只會獲得主節點）
```
kubectl get nodes --show-labels
```


50.標記節點（如果正在使用，則為 minikube）nodeName = nginxnode
```
kubectl label node minikube nodeName=nginxnode
```


51.建一個標籤為 nodeName = nginxnode 的 Pod 並將其部署在此節點上
```
kubectl run nginx --image=nginx --restart=Never --dry-run -o yaml > pod.yaml
// add the nodeSelector like below and create the pod
kubectl create -f pod.yaml
```