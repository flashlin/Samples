FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["WebAutoGrpcSample/WebAutoGrpcSample.csproj", "WebAutoGrpcSample/"]
RUN dotnet restore "WebAutoGrpcSample/WebAutoGrpcSample.csproj"
COPY . .
WORKDIR "/src/WebAutoGrpcSample"
RUN dotnet build "WebAutoGrpcSample.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "WebAutoGrpcSample.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "WebAutoGrpcSample.dll"]
