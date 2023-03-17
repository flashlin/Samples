FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["TestChatGptApi/TestChatGptApi.csproj", "TestChatGptApi/"]
RUN dotnet restore "TestChatGptApi/TestChatGptApi.csproj"
COPY . .
WORKDIR "/src/TestChatGptApi"
RUN dotnet build "TestChatGptApi.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "TestChatGptApi.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "TestChatGptApi.dll"]
