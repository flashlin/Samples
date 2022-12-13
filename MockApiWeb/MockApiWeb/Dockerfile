FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["MockApiWeb/MockApiWeb.csproj", "MockApiWeb/"]
RUN dotnet restore "MockApiWeb/MockApiWeb.csproj"
COPY . .
WORKDIR "/src/MockApiWeb"
RUN dotnet build "MockApiWeb.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MockApiWeb.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MockApiWeb.dll"]
