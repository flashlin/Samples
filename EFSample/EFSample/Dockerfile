FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["EFSample/EFSample.csproj", "EFSample/"]
RUN dotnet restore "EFSample/EFSample.csproj"
COPY . .
WORKDIR "/src/EFSample"
RUN dotnet build "EFSample.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "EFSample.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "EFSample.dll"]
