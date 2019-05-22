FROM microsoft/dotnet:2.2-aspnetcore-runtime AS base
WORKDIR /app
EXPOSE 80

FROM microsoft/dotnet:2.2-sdk AS build
WORKDIR /src
COPY ["test-etcd/test-etcd.csproj", "test-etcd/"]
RUN dotnet restore "test-etcd/test-etcd.csproj"
COPY . .
WORKDIR "/src/test-etcd"
RUN dotnet build "test-etcd.csproj" -c Release -o /app

FROM build AS publish
RUN dotnet publish "test-etcd.csproj" -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "test-etcd.dll"]