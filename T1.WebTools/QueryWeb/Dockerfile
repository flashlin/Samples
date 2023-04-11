FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["QueryWeb/QueryWeb.csproj", "QueryWeb/"]
COPY ["QueryKits/QueryKits.csproj", "QueryKits/"]
COPY ["QueryRazor/QueryRazor.csproj", "QueryRazor/"]
COPY ["T1.WebTools/T1.WebTools.csproj", "T1.WebTools/"]

RUN dotnet restore "QueryWeb/QueryWeb.csproj"

COPY . .

WORKDIR "/src/QueryWeb"
RUN dotnet build "QueryWeb.csproj" -c Release -o /app/build /p:ASPNETCORE_ENVIRONMENT=docker


# 
RUN apt-get update && apt-get install -y dos2unix
RUN dos2unix /src/QueryWeb/entrypoint.sh && chmod +x /src/QueryWeb/entrypoint.sh


FROM build AS publish
RUN dotnet publish "QueryWeb.csproj" -c Release -o /app/publish /p:UseAppHost=false


FROM base AS final
# Install sql server
RUN apt-get update && \
    apt-get install -y curl gnupg2 apt-transport-https wget software-properties-common && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    add-apt-repository "$(wget -qO- https://packages.microsoft.com/config/ubuntu/20.04/mssql-server-2022.list)" && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools unixodbc-dev mssql-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

ENV SA_PASSWORD=Passw0rd!
#CMD /opt/mssql/bin/sqlservr --accept-eula -error-log=/var/opt/mssql/log/errorlog & \
#    (echo $SA_PASSWORD | /opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P '' --query "SET NOCOUNT ON; SELECT 'SQL Server is up and running'")
#RUN /opt/mssql/bin/sqlservr --accept-eula -error-log=/var/opt/mssql/log/errorlog

# COPY ./Database /app/Database
#RUN chmod +x /app/Database/init.sh
#EXPOSE 1433

WORKDIR /app
COPY --from=publish /app/publish .

#
COPY --from=publish /src/QueryWeb/entrypoint.sh .


# ENTRYPOINT ["dotnet", "QueryWeb.dll"]
ENTRYPOINT ["./entrypoint.sh"]