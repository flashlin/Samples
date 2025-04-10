cd LocalSqlServer
./build-image.sh
cd ..
docker rm -f my_sql_server
docker run -d --name my_sql_server -p 4433:1433 -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrong@Passw0rd" my-sql-server 
echo docker logs my_sql_server
docker logs my_sql_server
