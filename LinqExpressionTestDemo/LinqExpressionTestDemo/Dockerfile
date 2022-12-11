FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["LinqExpressionTestDemo/LinqExpressionTestDemo.csproj", "LinqExpressionTestDemo/"]
RUN dotnet restore "LinqExpressionTestDemo/LinqExpressionTestDemo.csproj"
COPY . .
WORKDIR "/src/LinqExpressionTestDemo"
RUN dotnet build "LinqExpressionTestDemo.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "LinqExpressionTestDemo.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "LinqExpressionTestDemo.dll"]
