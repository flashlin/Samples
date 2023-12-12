Question: How to configure CDN domain restrictions for two or more paths in a Kubernetes YAML file for Ingress?

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

Question: How to configure environment variables for a dotnet container in Kubernetes using ConfigMap and ensure that dotnet's configuration can detect changes in the environment variables and reload accordingly?

Answer:
To configure environment variables for a container in a Kubernetes YAML file, you need to make some settings. 
To do this, you can use ConfigMap to define these environment variables and inject them into your container. 
Additionally, if your application is written in dotnet, 
you also need to ensure that dotnet's configuration can detect changes in the environment variables and reload accordingly.

To define environment variables, you need to create a ConfigMap. 
```
apiVersion: v1
kind: ConfigMap
metadata:
  name: your-setting-name
data:
  your-setting-name.json: |
    {
      "your_key_name": "blabla"
    }
```

In your deployment file (.gitlab-ci.yml), you need to add or run the following script.
```
script:
  - kubectl apply -f ./your-configMap.yml -n your_namespace_name
```

In deployment yml file, you need to add the following context.
For "DOTNET_USE_POLLING_FILE_WATCHER" environment variable, 
uses a polling file watcher instead of CoreFx's FileSystemWatcher. 
Used when watching files on network shares or Docker mounted volumes.
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: your_name
  namespace: your_namespace
spec:
  replicas: 1
  selector:
    matchLabels:
      name: your_name
  template:
    metadata:
      labels:
        name: your_name
    spec:
      containers:
        - name: your_name
          image: "your_image_name"
          volumeMounts:
            - mountPath: /etc/your-setting-name
              name: your-setting-name
          env:
            - name: "DOTNET_USE_POLLING_FILE_WATCHER"
              value: "true"
      volumes:
        - name: your-setting-name
          projected:
            sources:
              - configMap:
                  name: your-setting-name
```

In program.cs file, you need to add the following code.
```
public static IHostBuilder CreateHostBuilder(string[] args) =>
    Host.CreateDefaultBuilder(args)
        .ConfigureAppConfiguration((hostingContext, config) =>
        {
            config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddJsonFile($"appsettings.{Env}.json", optional: true, reloadOnChange: true)
                .AddJsonFile("/etc/your-setting-name/your-setting-name.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables();
        });
```

In startup.cs file, you need to add the following code.
```
public void ConfigureServices(IServiceCollection services)
{
  services.Configure<YourConfig>(Configuration);
}
```

In application code, you can use the following code to get the configuration.
```
public class YourObject
{
  public YourObject(IOptionsMonitor<YourConfig> config)
  {
    var conf = config.CurrentValue;
  }
}
```

Question: When you run docker ps in WSL or a Linux shell, you encounter the following error message. 
How to resolve it?
`Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock`
Answer:
Execute the following command, then restart WSL or Linux.
```bash
sudo usermod -aG docker $USER
```

If you don't want to restart WSL or Linux but just temporarily grant permission, you can execute the following command to immediately use Docker.
```bash
sudo chmod 666 /var/run/docker.sock
```

Question: How to check the CPU of POD?
Answer: To check POD CPU/MEM usage with cmd
```
kubectl top pods -n b2c  | grep <your-project-name>
```

Question: How to install Chocolatey?
Answer: Follow the steps at https://chocolatey.org/install to install Chocolatey.

Question: 
Answer:
* Follow the steps at https://chocolatey.org/install to install Chocolatey.
* Install kubernetes-cli
```bash
choco install kubernetes-cli
```
After installation, you should be able to use kubectl.
* Download Kubernetes cluster token from Rancher WebSite.
Upon opening Rancher, you will see many clusters there.
|State |Name |Provider |Kubernetes Version |CPU |Memory |Pods
|--|--|--|--|--|--|--
|Active |host-prod |Imported |v1.26.7 |510.16 cores |1.88 TiB |18/800
|Active |host-staging |Imported |v1.26.7 |222.955 cores |816 GiB |421/770
|Active |host-uat |Imported |v1.26.7 |286.65 cores |1.02 TiB |892/990

Access each of them and download the kubeconfig files, then place them in a separate folder (c:\k8s-certs).

Try to explore Kubernetes cluster by below command
```
kubectl get pod -n b2c --kubeconfig='C:/k8s-certs/k8s-stg.yaml'
```

Question: How to merge kubeconfig files?
Question: How to merged kube config file with all the cluster cert? 
Question: How can I merge the kubeconfig file with all the cluster certificates?
Answer: By WSL
```WSL
KUBECONFIG=./host-uat-gke.yaml:./host-prod-gke.yaml:./host-staging-gke.yaml:./k8s-prod-backup.yaml:./k8s-prod.yaml:./k8s-stg.yaml:./mt-host-prod.yaml:./mt-host-stag-gke.yaml kubectl config view --flatten > config
```
By Bash
```bash
KUBECONFIG=.\\host-uat-gke.yaml:.\\host-prod-gke.yaml:.\\host-staging-gke.yaml:.\\k8s-prod-backup.yaml:.\\k8s-prod.yaml:.\\k8s-stg.yaml:.\\mt-host-prod.yaml:.\\mt-host-stag-gke.yaml kubectl config view --flatten > config
```

Question: Easy way to switch between different clusters.
Question: I have multiple kubeconfig files. How can I easily switch between different clusters?
Answer:
* Merge the kubeconfig file with all the cluster certificates, Put the outputed file config to ~/.kube
* Restart your Powershell and try to type command with different cluster name  
```bash
kubectl config use-context k8s-stg
kubectl get pod -n b2c-payment
```

Install kubectx and fzf for fast switching. (https://github.com/ahmetb/kubectx)
```bash
choco install kubectx
choco install fzf
```
Enter kubectx in your PowerShell. You'll be able to view all the clusters and select the desired one.


Question: Enable kubectl autocompletion with alias k 
Answer:
```powershell
# Enable kubectl autocompletion with alias k
(kubectl completion powershell)  -replace "'kubectl'", "'k'" | Out-String | Invoke-Expression
```

```powershell
Set-Alias k kubectl
Set-Alias kx kubectx
Set-Alias kns kubens
```

```bash
alias k=kubectl
alias kx=kubectx
alias kns=kubens
```