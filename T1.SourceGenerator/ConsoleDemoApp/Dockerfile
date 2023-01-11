FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["ConsoleDemoApp/ConsoleDemoApp.csproj", "ConsoleDemoApp/"]
RUN dotnet restore "ConsoleDemoApp/ConsoleDemoApp.csproj"
COPY . .
WORKDIR "/src/ConsoleDemoApp"
RUN dotnet build "ConsoleDemoApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ConsoleDemoApp.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ConsoleDemoApp.dll"]
