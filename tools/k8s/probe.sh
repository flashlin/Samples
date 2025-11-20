#!/bin/bash

set -e

export KUBECONFIG=~/Downloads/rke2-stg.yaml

NAMESPACE="b2c"
CONTAINER_NAME="runtime-monitor"
AUTH_TOKEN="330a1646-4a53-4b0c-8c57-1d7a4f5eda4a"
LOCAL_PORT=52323

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed"
    exit 1
fi

if [ ! -f "$KUBECONFIG" ]; then
    echo "Error: KUBECONFIG file not found: $KUBECONFIG"
    exit 1
fi

echo "=========================================="
echo "Probe Test Script"
echo "=========================================="
echo "KUBECONFIG: $KUBECONFIG"
echo "Namespace: $NAMESPACE"
echo "=========================================="
echo ""

echo "Finding running tracking-api pods..."
POD_NAME=$(kubectl get pods -n $NAMESPACE -l name=tracking-api --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "Error: No running tracking-api pod found"
    echo ""
    echo "Available pods:"
    kubectl get pods -n $NAMESPACE -l name=tracking-api
    exit 1
fi

echo "Found pod: $POD_NAME"
echo ""

echo "Checking if container $CONTAINER_NAME is running..."
CONTAINER_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")].ready}' 2>/dev/null)

if [ "$CONTAINER_STATUS" != "true" ]; then
    echo "Warning: Container $CONTAINER_NAME is not ready"
    echo "Container status:"
    kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")]}' | jq '.' 2>/dev/null || kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.containerStatuses[?(@.name=="'$CONTAINER_NAME'")]}'
    echo ""
fi

echo "Setting up port-forward..."
echo "Forwarding port $LOCAL_PORT to pod $POD_NAME:52323"
kubectl port-forward -n $NAMESPACE $POD_NAME $LOCAL_PORT:52323 > /dev/null 2>&1 &
PORT_FORWARD_PID=$!

sleep 2

if ! kill -0 $PORT_FORWARD_PID 2>/dev/null; then
    echo "Error: Failed to establish port-forward"
    exit 1
fi

echo "Port-forward established (PID: $PORT_FORWARD_PID)"
echo ""

echo "Testing /processes endpoint..."
echo "Command: curl -v -H \"Authorization: Bearer $AUTH_TOKEN\" http://localhost:$LOCAL_PORT/processes"
echo ""

curl -v -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:$LOCAL_PORT/processes

CURL_EXIT_CODE=$?

echo ""
echo "Cleaning up port-forward..."
kill $PORT_FORWARD_PID 2>/dev/null || true
wait $PORT_FORWARD_PID 2>/dev/null || true

echo ""
echo "=========================================="
if [ $CURL_EXIT_CODE -eq 0 ]; then
    echo "Probe test completed successfully!"
else
    echo "Probe test failed with exit code: $CURL_EXIT_CODE"
fi
echo "=========================================="

exit $CURL_EXIT_CODE

