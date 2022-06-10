#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["WebSite2/WebSite2.csproj", "WebSite2/"]
RUN dotnet restore "WebSite2/WebSite2.csproj"
COPY . .
WORKDIR "/src/WebSite2"
RUN dotnet build "WebSite2.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "WebSite2.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "WebSite2.dll"]