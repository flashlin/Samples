FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["QueryApp/QueryApp.csproj", "QueryApp/"]
RUN dotnet restore "QueryApp/QueryApp.csproj"
COPY . .
WORKDIR "/src/QueryApp"
RUN dotnet build "QueryApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "QueryApp.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "QueryApp.dll"]
