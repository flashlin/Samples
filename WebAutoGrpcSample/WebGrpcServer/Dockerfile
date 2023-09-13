FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["WebGrpcServer/WebGrpcServer.csproj", "WebGrpcServer/"]
RUN dotnet restore "WebGrpcServer/WebGrpcServer.csproj"
COPY . .
WORKDIR "/src/WebGrpcServer"
RUN dotnet build "WebGrpcServer.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "WebGrpcServer.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "WebGrpcServer.dll"]
