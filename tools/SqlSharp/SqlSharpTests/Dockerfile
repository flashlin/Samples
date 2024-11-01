FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Debug

WORKDIR /app
COPY ["SqlSharpTests/SqlSharpTests.csproj", "SqlSharpTests/"]
COPY ["SqlSharpLit/SqlSharpLit.csproj", "SqlSharpLit/"]
RUN dotnet restore "SqlSharpTests/SqlSharpTests.csproj"
COPY . .

RUN dotnet build "SqlSharpTests.csproj" -c $BUILD_CONFIGURATION -o /app/build

WORKDIR /app/SqlSharpTests
ENTRYPOINT ["dotnet", "test", "--no-build", "--logger:trx;LogFileName=TestResults.trx", "SqlSharpTests.dll"]
