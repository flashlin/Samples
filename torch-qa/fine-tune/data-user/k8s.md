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