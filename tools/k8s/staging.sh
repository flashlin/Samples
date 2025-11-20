#!/bin/bash

set -e

export KUBECONFIG=~/Downloads/rke2-stg.yaml

K8S_DEPLOYMENT="deployment/tracking-api"
ROLLOUT_STATUS_TIMEOUT="2m"
CLUSTER="staging"
NAMESPACE="b2c"

if [ -z "$1" ]; then
    echo "Usage: $0 <DOCKER_IMAGE_WITH_TAG>"
    echo "Example: $0 asia.gcr.io/registry-b45d6b28/twdc/tracking.api:latest"
    exit 1
fi

DOCKER_IMAGE_WITH_TAG="$1"

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

if ! command -v kustomize &> /dev/null; then
    echo "Error: kustomize is not installed"
    exit 1
fi

if [ ! -f "$KUBECONFIG" ]; then
    echo "Error: KUBECONFIG file not found: $KUBECONFIG"
    exit 1
fi

echo "=========================================="
echo "Staging Deployment Script"
echo "=========================================="
echo "KUBECONFIG: $KUBECONFIG"
echo "Docker Image: $DOCKER_IMAGE_WITH_TAG"
echo "Cluster: $CLUSTER"
echo "Namespace: $NAMESPACE"
echo "=========================================="
echo ""

cd kubernetes-manifest/overlays/$CLUSTER

echo "Setting image to: $DOCKER_IMAGE_WITH_TAG"
kustomize edit set image $DOCKER_IMAGE_WITH_TAG

echo ""
echo "Building and applying manifests..."
kustomize build . | kubectl apply -f -

echo ""
echo "Waiting for deployment rollout..."
kubectl rollout status $K8S_DEPLOYMENT -n $NAMESPACE --timeout=$ROLLOUT_STATUS_TIMEOUT

echo ""
echo "=========================================="
echo "Deployment Status"
echo "=========================================="
kubectl get deployment tracking-api -n $NAMESPACE

echo ""
echo "=========================================="
echo "Deployment Details"
echo "=========================================="
kubectl describe deployment tracking-api -n $NAMESPACE

echo ""
echo "=========================================="
echo "Pods Status"
echo "=========================================="
kubectl get pods -n $NAMESPACE -l name=tracking-api

echo ""
echo "=========================================="
echo "Pods Details"
echo "=========================================="
kubectl describe pods -n $NAMESPACE -l name=tracking-api

echo ""
echo "=========================================="
echo "Tracking-API Container Logs (last 50 lines)"
echo "=========================================="
kubectl logs -n $NAMESPACE -l name=tracking-api -c tracking-api --tail=50 2>/dev/null || echo "No logs available yet"

echo ""
echo "=========================================="
echo "Runtime-Monitor Container Logs (last 50 lines)"
echo "=========================================="
kubectl logs -n $NAMESPACE -l name=tracking-api -c runtime-monitor --tail=50 2>/dev/null || echo "No logs available yet"

echo ""
echo "=========================================="
echo "Deployment completed!"
echo "=========================================="

