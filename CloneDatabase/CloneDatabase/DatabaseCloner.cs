using LanguageExt;
using LanguageExt.Common;
using Microsoft.Extensions.Options;

namespace CloneDatabase;

public class DatabaseCloner
{
    private DbConfig _dbConfig;
    private SqlDb _sourceDb = null!;
    private readonly ISqlScriptBuilder _scriptBuilder;
    private SqlDb _targetDb = null!;

    public DatabaseCloner(IOptions<DbConfig> dbConfig, ISqlScriptBuilder scriptBuilder)
    {
        _scriptBuilder = scriptBuilder;
        _dbConfig = dbConfig.Value;
        SetServer(_dbConfig);
    }

    public DatabaseCloner SetServer(DbConfig dbConfig)
    {
        _dbConfig = dbConfig;
        _sourceDb = new SqlDb(_dbConfig.SourceServer);
        _targetDb = new SqlDb(_dbConfig.TargetServer);
        return this;
    }

    public void Clone()
    {
        var errors = new List<Exception>();
        foreach (var dbInfo in _sourceDb.QueryDatabases())
        {
            if (dbInfo.Name.Equals("tempdb", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            Console.WriteLine($"{dbInfo.Name} {dbInfo.CreateDate:yyyy-MM-dd}");

            var tableNames = _sourceDb.QueryTableNames(dbInfo.Name);
            foreach (var tableName in tableNames)
            {
                Console.WriteLine($"\t{tableName}");
                CloneTable(new CloneTableReq
                {
                    SourceDbName = dbInfo.Name,
                    SourceTableName = tableName,
                    TargetDbName = dbInfo.Name
                });
                // var fields = _sourceDb.QueryFields(dbInfo.Name, tableName);
                // foreach (var field in fields)
                // {
                //     Console.WriteLine($"\t\t{field.Name} {field.IsNullable}");
                // }
            }
        }
    }

    public void CloneTable(CloneTableReq req)
    {
        var targetTable = _targetDb.QueryTable(req.TargetDbName, req.SourceTableName);
        if (targetTable == null)
        {
            var sourceTable = _sourceDb.QueryTable(req.SourceDbName, req.SourceTableName);
            var createTableScript = _scriptBuilder.CreateTableSql(sourceTable!);
            ExecuteSql(_targetDb, req.TargetDbName, createTableScript);
            return;
        }

        var dropTableScript = _scriptBuilder.DeleteTableSql(req.SourceTableName);
        ExecuteSql(_targetDb, req.TargetDbName, dropTableScript);
    }

    private void ExecuteSql(SqlDb db, string dbName, string sql)
    {
        db.Execute($"use [{dbName}];" + sql);
    }
}