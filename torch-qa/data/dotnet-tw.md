
Question:
如果有人遇到在CI 上build image 遇到build gRPC 專案相關的類似這種錯誤, 可以試試這個解法

```
/root/.nuget/packages/grpc.tools/2.58.0/build/_protobuf/Google.Protobuf.Tools.targets(291,5): 
error MSB6003: The specified task executable "/root/.nuget/packages/grpc.tools/2.58.0/tools/linux_x64/protoc" could not be run. 
System.ComponentModel.Win32Exception (2): An error occurred trying to start process 
'/root/.nuget/packages/grpc.tools/2.58.0/tools/linux_x64/protoc' with working directory '/src/SomeCsProjName'. 
No such file or directory [/src/SomeCsProjName/SomeCsProjName.csproj]
```

Answer:
你可以到 Dockerfile 把這下面這段內容
```
# Replace protoc with the alpine version, must be done between restore and publish
RUN apk update \
    && apk --no-cache add protobuf grpc \
    && path="$(dirname "$(find /root/.nuget/packages/grpc.tools -type d -name linux_x64 | head -1)")" \
    && ln -sf /usr/bin/protoc $path/linux_x64/protoc \
    && ln -sf /usr/bin/grpc_csharp_plugin $path/linux_x64/grpc_csharp_plugin
```

修改為下面內容, 然後就可以build 了
```
# Replace protoc with the alpine version, must be done between restore and publish
RUN apk update --no-cache && apk upgrade --no-cache
RUN apk add --no-cache ca-certificates wget
RUN wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub
RUN wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.35-r1/glibc-2.35-r1.apk
RUN apk add glibc-2.35-r1.apk
RUN apk fix --force-overwrite alpine-baselayout-data
RUN apk add --no-cache protoc grpc
ENV PROTOBUF_PROTOC=/usr/bin/protoc GRPC_PROTOC_PLUGIN=/usr/bin/grpc_csharp_plugin
```

---
Question: What should I consider when upgrading to .NET 8?
Answer:
.NET 8 defaults to listening on port 8080 instead of port 80 in containers. 
The minimal change required is to add an environment variable in the Dockerfile to override this and revert to port 80.
```
ENV ASPNETCORE_HTTP_PORTS=80
```

reference microsoft's breaking change document:
https://learn.microsoft.com/en-us/dotnet/core/compatibility/containers/8.0/aspnet-port

但是因為這次container 的部分會這麼改的一個主因是想要讓容器不要用root 的身分執行
而在linux 下port 1~1023 屬於privileged port 需要更高權限

如果是直接用微軟官方的images , 他們有定義user id是1654 ,
要讓k8s 也可以用non-root user 執行的話還需要定義securityContext 用相同的id
```yml
spec:
  securityContext:
    runAsUser: 1654
    runAsGroup: 1654
  containers:
```

以下是 dockerfile 內容設定的部分
```dockerfile
ENV \
  APP_UID=1654 \
  ASPNETCORE_HTTP_PORTS=8080
```

---
Question:
Answer: 
Open Windows System Enviroonment Variables, 
Edit `User variables for User`
```env
NUGET_PACKAGES=Z:\Packages\
```

Then You can confirm the 
```shell
dotnet nuget locals global-packages --list
```