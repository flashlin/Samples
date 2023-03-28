FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["ClientConsoleApp/ClientConsoleApp.csproj", "ClientConsoleApp/"]
RUN dotnet restore "ClientConsoleApp/ClientConsoleApp.csproj"
COPY . .
WORKDIR "/src/ClientConsoleApp"
RUN dotnet build "ClientConsoleApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ClientConsoleApp.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ClientConsoleApp.dll"]
