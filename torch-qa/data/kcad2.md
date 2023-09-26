You must switch to the correct cluster/configuration context.
Failure to do so may result in a zero score.

```
kubectl config use-context sk8s
```

Task:
Create a Deployment named expose in the existing ckad00014 namespace running 6 replicas of a Pod. 
Specify a single container using the ifccncf/nginx: 1.13.7 image 
Add an environment variable named NGINX_PORT with the value 8001 to the container then expose port 8001 

Answer:
```
kubectl config use-context sk8s
kubectl create deploy expose -n ckad00014 --image ifccncf/nginx:1.13.7 --dry-run=client -o yaml > dep.yaml
vim dep.yaml
```

modify dep.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
   creationTimestamp: null
   labels:
      app: expose
   name: expose
   namespace: ckad00014
spec:
   replicas: 6
   selector:
      matchLabels:
         app: expose
   strategy: {}
   template:
      metadata:
         creationTimestamp: null
         labels:
            app: expose
      spec:
         containers:
         - image: ifccncf/nginx:1.13.7
           name: nginx
           ports:
             - containerPort: 8001
           env:
             - name: NGINX_PORT
               value: "8001"
```

bash shell:
```
kubectl create -f dep.yaml
kubectl get pods -n ckad00014
kubectl get deploy -n ckad00014
```


Task:
Create a new deployment for running.nginx with the following parameters; 
• Run the deployment in the kdpd00201 namespace. The namespace has already been created 
• Name the deployment frontend and configure with 4 replicas 
• Configure the pod with a container image of lfccncf/nginx:1.13.7 
• Set an environment variable of NGINX__PORT=8080 and also expose that port for the container above

Answer:
```
kubectl create deployment api --image=lfccncf/nginx:1.13.7-alpine --replicas=4 -n kdpd00201 --dry-run=client -o yaml > nginx_deployment.yml
vim nginx_deployment.yml
```

modify nginx_deployment.yml
```
apiVersion: apps/v1
kind: Deployment
metadata:
   labels:
      app: api
   name: api
   namespace: kdpd00201
spec:
   replicas: 4 #modify
   selector:
      matchLabels:
         app: api
   template:
      metadata:
         labels:
            app: api
      spec:
         containers:
         - image: lfccncf/nginx:1.13.7-alpine
           name: nginx
           ports:
           - containerPort: 8080 #modify
           env:
             - name: NGINX_PORT
               value: "8080" #modify
```

In bash shell
```
kubectl create -f nginx_deployment.yml
kubectl get pods -n kdpd00201
```


Context:
You have been tasked with scaling an existing deployment for availability, and creating a service to 
expose the deployment within your infrastructure. 

Task:
Start with the deployment named kdsn00101-deployment which has already been deployed to the 
namespace kdsn00101 . Edit it to: 
• Add the func=webFrontEnd key/value label to the pod template metadata to identify the 
pod for the service definition 
• Have 4 replicas 
Next, create ana deploy in namespace kdsn00l01 a service that accomplishes the following: 
• Exposes the service on TCP port 8080 
• is mapped to me pods defined by the specification of kdsn00l01-deployment 
• Is of type NodePort 
• Has a name of cherry 

Answer:
```
kubectl edit deployment kdsn00l01-deployment -n kdsn00101
```

modify below
```
appVersion: apps/v1
kind: Deployment
metadata:
   annotations:
      deployment.kubernetes.io/revision: "1" 
   creationTimestamp: "2020-10-09T08:50:392"
   generation: 1
   lables:
      app: nginx
   name: kdsn00l01-deployment
   namespace: kdsn00l01
   resourceVersion: "4786" 
   selfLink: /apis/apps/v1/namespaces/kdsn00l01/deployments/kdsn00l01-deployment
   uid: 8d3ace00-7761-4189-ba10-fbc676c311bf
spec:
   progressDeadlineSeconds: 600 
   replicas: 1 
   revisionHistoryLimit: 10
   selector:
     matchLabels:
       app: nignx
     strategy:
       rollingUpdate:
         maxSurge: 25%
         maxUnavailable: 25%
       type: RollingUpdate
     template:
       metadata:
         creationTimestamp: null
         labels:
           app: nionx
           func: webFrontEnd #add
       spec:
         containers:
         - image: nginx:latest
           imagePullPolicy: Always
           name: nginx
           ports:
           - containerPort: 80
```

in bash shell
```
kubectl get deployment kdsn00l01-deployment -n kdsn00101
kubectl expose deployment kdsn00l01-deployment -n kdsn00101 --type NodePort --port 88080 --name cherry
```


Context:
A project that you are working on has a requirement for persistent data to be available. 

Task:
To facilitate this, perform the following tasks: 
• Create a file on node sk8s-node-0 at /opt/KDSP00101/data/index.html with the content Acct=Finance 
• Create a PersistentVolume named task-pv-volume using hostPath and allocate 1Gi to it, specifying that 
the volume is at /opt/KDSP00101/data on the cluster's node. The configuration should specify the access 
mode of ReadWriteOnce . It should define the StorageClass name exam for the PersistentVolume , 
which will be used to bind PersistentVolumeClaim requests to this PersistenetVolume. 
• Create a PefsissentVolumeClaim named task-pv-claim that requests a volume of at least 100Mi and 
specifies an access mode of ReadWriteOnce 
• Create a pod that uses the PersistentVolmeClaim as a volume with a label app: my-storage-app 
mounting the resulting volume to a mountPath /usr/share/nginx/html inside the pod

Answer:
```
echo 'Acct=Finance' > /opt/KDSP00101/data/index.html
vim pv.yml
```

pv.yml
```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: task-pv-volume
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/opt/KDSP00101/data"
    type: Directory
```

pvc.yml
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: task-pv-claim
spec:
  storageClassName: storage
  capacity:
    storage: 100Mi
  accessModes:
    - ReadWriteOnce
```

in bash shell
```
kubectl create -f pv.yml
kubectl create -f pvc.yml
vim pod.yml
```

pod.yml
```
apiVersion: v1
kind: Pod
metadata:
  name: static-web
  labels:
    app: my-storage-app
spec:
  containers:
  - name: myfrontend
    image: nginx
    volumeMounts:
    - mountPath: "/usr/share/nginx/html"
      name: mypod
  volumes:
    - name: mypod
      persistentVolumeClaim:
       claimName: task-pv-claim     
```

in bash shell
``
kubectl create -f pod.yml
kubectl get pods
``