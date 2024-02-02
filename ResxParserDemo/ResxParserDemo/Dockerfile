FROM mcr.microsoft.com/dotnet/runtime:8.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["ResxParserDemo/ResxParserDemo.csproj", "ResxParserDemo/"]
RUN dotnet restore "ResxParserDemo/ResxParserDemo.csproj"
COPY . .
WORKDIR "/src/ResxParserDemo"
RUN dotnet build "ResxParserDemo.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ResxParserDemo.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ResxParserDemo.dll"]
