﻿FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["CloneDatabase/CloneDatabase.csproj", "CloneDatabase/"]
RUN dotnet restore "CloneDatabase/CloneDatabase.csproj"
COPY . .
WORKDIR "/src/CloneDatabase"
RUN dotnet build "CloneDatabase.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "CloneDatabase.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "CloneDatabase.dll"]
