#!/bin/bash
echo initialize database
#do this in a loop because the timing for when the SQL instance is ready is indeterminate
for i in {1..50};
do
    /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P Passw0rd! -d master -i init.sql
    if [ $? -eq 0 ]
    then
        echo ""
        echo "--------------------"
        echo "init.sql completed"
        break
    else
        echo "not ready yet..."
        sleep 1
    fi
done

