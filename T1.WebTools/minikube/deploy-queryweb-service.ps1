kubectl apply -f queryweb-service.yaml
# 確認是否成功
kubectl get service queryweb-service
# 獲取 IP
minikube ip
