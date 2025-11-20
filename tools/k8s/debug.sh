#!/bin/bash

set -e

export KUBECONFIG=~/Downloads/rke2-stg.yaml

NAMESPACE="b2c"
CONTAINER_NAME="runtime-monitor"

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

if [ ! -f "$KUBECONFIG" ]; then
    echo "Error: KUBECONFIG file not found: $KUBECONFIG"
    exit 1
fi

echo "=========================================="
echo "Monitor Debug Script"
echo "=========================================="
echo "KUBECONFIG: $KUBECONFIG"
echo "Namespace: $NAMESPACE"
echo "=========================================="
echo ""

echo "Finding tracking-api pods with unready monitor container..."
ALL_PODS=$(kubectl get pods -n $NAMESPACE -l name=tracking-api -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')

if [ -z "$ALL_PODS" ]; then
    echo "No tracking-api pods found"
    exit 1
fi

POD_NAME=""
for pod in $ALL_PODS; do
    READY_STATUS=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")].ready}' 2>/dev/null)
    if [ "$READY_STATUS" != "true" ]; then
        POD_NAME=$pod
        break
    fi
done

if [ -z "$POD_NAME" ]; then
    echo "No unready monitor container found. Checking all pods..."
    echo ""
    kubectl get pods -n $NAMESPACE -l name=tracking-api -o wide
    echo ""
    echo "All monitor containers appear to be ready."
    exit 0
fi

echo "Found unready pod: $POD_NAME"
echo ""

echo "=========================================="
echo "Pod Details - Readiness Probe Info"
echo "=========================================="
kubectl describe pod $POD_NAME -n $NAMESPACE | grep -A 20 "Readiness" || echo "No Readiness probe info found"

echo ""
echo "=========================================="
echo "Readiness Probe Failure Events"
echo "=========================================="
kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$POD_NAME --sort-by='.lastTimestamp' | grep -i "readiness\|unhealthy" || echo "No readiness/unhealthy events found"

echo ""
echo "=========================================="
echo "All Recent Events for Pod"
echo "=========================================="
kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$POD_NAME --sort-by='.lastTimestamp' | tail -20

echo ""
echo "=========================================="
echo "Container Status"
echo "=========================================="
kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")]}' | jq '.' 2>/dev/null || kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")]}'

echo ""
echo "=========================================="
echo "Debug completed!"
echo "=========================================="

