FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Debug

WORKDIR /app
COPY ["SqlSharpTests/SqlSharpTests.csproj", "SqlSharpTests/"]
COPY ["SqlSharpLit/SqlSharpLit.csproj", "SqlSharpLit/"]
RUN dotnet restore "SqlSharpTests/SqlSharpTests.csproj"
COPY . .

RUN dotnet build "SqlSharpTests/SqlSharpTests.csproj" -c $BUILD_CONFIGURATION -o /app

WORKDIR /app
# ENTRYPOINT ["dotnet", "test", "--logger:trx;LogFileName=TestResults.trx", "SqlSharpTests.dll"]
# CMD ["sh", "-c", "dotnet test --logger:trx;LogFileName=TestResults.trx && tail -f /dev/null"]
CMD ["sh", "-c", "dotnet test --logger 'trx;LogFileName=/app/TestResults.trx'"]
