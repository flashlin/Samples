#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["DemoWeb/DemoWeb.csproj", "DemoWeb/"]
RUN dotnet restore "DemoWeb/DemoWeb.csproj"
COPY . .
WORKDIR "/src/DemoWeb"
RUN dotnet build "DemoWeb.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "DemoWeb.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "DemoWeb.dll"]