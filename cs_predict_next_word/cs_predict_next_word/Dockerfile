FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["cs_predict_next_word/cs_predict_next_word.csproj", "cs_predict_next_word/"]
RUN dotnet restore "cs_predict_next_word/cs_predict_next_word.csproj"
COPY . .
WORKDIR "/src/cs_predict_next_word"
RUN dotnet build "cs_predict_next_word.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "cs_predict_next_word.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "cs_predict_next_word.dll"]
