# Set docker env
& minikube docker-env | Invoke-Expression

& curl https://192.168.2.10:8443 --cert C:\Users\flash\.minikube\apiserver.crt --key C:\Users\flash\.minikube\apiserver.key --cacert C:\Users\flash\.minikube\ca.crt

# build image
& .\create.ps1
# run in minikube
kubectl run vue-dev --image=vue-sample --image-pull-policy=Never --port=8080 deployment "hello-vue"
#
kubectl expose deployment "hello-vue" --type=NodePort service "hello-vue-service" exposed

minikube service "hello-vue-service" --url

kubectl get pods

kubectl create -f vue-sample-pod.yaml