FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["RecursiveDescentParserDemo/RecursiveDescentParserDemo.csproj", "RecursiveDescentParserDemo/"]
RUN dotnet restore "RecursiveDescentParserDemo/RecursiveDescentParserDemo.csproj"
COPY . .
WORKDIR "/src/RecursiveDescentParserDemo"
RUN dotnet build "RecursiveDescentParserDemo.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "RecursiveDescentParserDemo.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "RecursiveDescentParserDemo.dll"]
