FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["CallChatGpt/CallChatGpt.csproj", "CallChatGpt/"]
RUN dotnet restore "CallChatGpt/CallChatGpt.csproj"
COPY . .
WORKDIR "/src/CallChatGpt"
RUN dotnet build "CallChatGpt.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "CallChatGpt.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "CallChatGpt.dll"]
