#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["QueryWeb/QueryWeb.csproj", "QueryWeb/"]
COPY ["QueryKits/QueryKits.csproj", "QueryKits/"]
COPY ["QueryRazor/QueryRazor.csproj", "QueryRazor/"]
COPY ["T1.WebTools/T1.WebTools.csproj", "T1.WebTools/"]

RUN dotnet restore "QueryWeb/QueryWeb.csproj"

COPY . .

WORKDIR "/src/QueryWeb"
RUN dotnet build "QueryWeb.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "QueryWeb.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final

# Install necessary packages
RUN apt-get update && \
    apt-get install -y curl gnupg2 apt-transport-https && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools unixodbc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "QueryWeb.dll"]