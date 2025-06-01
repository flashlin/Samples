# 使用 .NET 9.0 SDK 作為建置階段
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY . .
RUN dotnet restore VimSharpApp.csproj
RUN dotnet publish VimSharpApp.csproj -c Release -o /app/publish

# 使用 .NET 9.0 Runtime 作為執行階段
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080
ENTRYPOINT ["dotnet", "VimSharpApp.dll"] 