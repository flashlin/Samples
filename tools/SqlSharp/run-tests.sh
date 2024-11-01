docker build -t nunit-sql-sharp-tests -f SqlSharpTests/Dockerfile .
docker run --rm nunit-sql-sharp-tests