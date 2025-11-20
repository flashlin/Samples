export KUBECONFIG=~/Downloads/rke2-prod-a.yaml
#export KUBECONFIG=~/Downloads/rke2-prod-b.yaml
#kubectl get pods -n b2c
kubectl get deployment otel-deployment-collector -n b2c
#kubectl get pods -n b2c | grep otel