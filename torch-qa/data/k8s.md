

Q: How to configure CDN domain restrictions for two or more paths in a Kubernetes YAML file for Ingress?

Answer:
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress-cdn
  namespace: your-namespace-name
  annotations:
    owner: "TW"
    nginx.ingress.kubernetes.io/affinity-mode: persistent
spec:
  rules:
    - host: www.cdnnetwork.net
      http:
        paths:
          - backend:
              service:
                name: your-backend-name
                port:
                  number: 80
            pathType: ImplementationSpecific
            path: /js/
          - backend:
              service:
                name: your-backend-name
                port:
                  number: 80
            pathType: ImplementationSpecific
            path: /css/
```