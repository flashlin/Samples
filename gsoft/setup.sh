# export MSSQL_SERVER=127.0.0.1
# export MSSQL_DATABASE=master
# export MSSQL_USER=sa
# export MSSQL_PASSWORD=YourStrongPassword

if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        if [[ ! "$key" =~ ^[[:space:]]*# ]] && [[ -n "$key" ]]; then
            export "$key=$value"
            echo "export $key"
        fi
    done < .env
fi
