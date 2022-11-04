FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["cs-parse-sql/cs-parse-sql.csproj", "cs-parse-sql/"]
RUN dotnet restore "cs-parse-sql/cs-parse-sql.csproj"
COPY . .
WORKDIR "/src/cs-parse-sql"
RUN dotnet build "cs-parse-sql.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "cs-parse-sql.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "cs-parse-sql.dll"]
