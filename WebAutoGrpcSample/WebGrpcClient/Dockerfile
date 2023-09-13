FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["WebGrpcClient/WebGrpcClient.csproj", "WebGrpcClient/"]
RUN dotnet restore "WebGrpcClient/WebGrpcClient.csproj"
COPY . .
WORKDIR "/src/WebGrpcClient"
RUN dotnet build "WebGrpcClient.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "WebGrpcClient.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "WebGrpcClient.dll"]
