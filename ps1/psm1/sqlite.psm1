Add-Type -Path "$($env:psm1HOME)/../sqlite/System.Data.SQLite.dll"
$path = "$($env:psm1HOME)/../sqlite"
if ($env:PATH -notlike "*$path*") {
    $env:PATH += ";$path"
}

class Sqlite {
    [string]$connectionString = "Data Source=$databasePath;Version=3;"
    $connection = $null
    
    Sqlite([string]$dbFile) {
        $this.connectionString = "Data Source=$dbFile;Version=3;"
        $this.connection = New-Object -TypeName System.Data.SQLite.SQLiteConnection
        $this.connection.ConnectionString = $this.connectionString
        $this.connection.Open()
    }
    
    [void] ExecuteNonQuery([string] $sql) {
        $command = $this.connection.CreateCommand()
        $command.CommandText = $sql
        $affectedRows = $command.ExecuteNonQuery()
    }

    [void] FetchQuery([string]$query,        
        [scriptblock]$processRow) {
        $command = $this.connection.CreateCommand()
        $command.CommandText = $query
        $reader = $command.ExecuteReader()
        $schemaTable = $reader.GetSchemaTable()
        $columnNames = @()
        foreach ($row in $schemaTable.Rows) {
            $columnName = $row["ColumnName"]
            $columnNames += $columnName
        }
        while ($reader.Read()) {
            $row = @{}
            foreach ($columnName in $columnNames) {
                $columnValue = $reader[$columnName]
                AddObjectProperty $row $columnName $columnValue
                # $row | Add-Member -MemberType NoteProperty -Name $columnName -Value $columnValue
            }
            # Write-Host (DumpObject $row)
            $processRow.Invoke($row)
        }
    }

    [array] ExecuteQuery([string]$query) {
        # Write-Host "execute Query $query"
        $command = $this.connection.CreateCommand()
        $command.CommandText = $query
        $reader = $command.ExecuteReader()
        $schemaTable = $reader.GetSchemaTable()
        $columnNames = @()
        foreach ($row in $schemaTable.Rows) {
            $columnName = $row["ColumnName"]
            $columnNames += $columnName
        }
        $result = @()
        while ($reader.Read()) {
            $row = @{}
            foreach ($columnName in $columnNames) {
                $columnValue = $reader[$columnName]
                # Write-Host "c $columnValue"
                $row | Add-Member -MemberType NoteProperty -Name $columnName -Value $columnValue
            }
            $result += $row
        }
        return $result
    }

    [void] Close() {
        $this.connection.Close()
    }
}

function new-sqlite {
    param( 
        [string]$dbFile
    )
    return [sqlite]::new($dbFile)
}

Export-ModuleMember -Function *