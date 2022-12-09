using System.Data.SqlClient;
using Dapper;
using LanguageExt;
using LanguageExt.Common;
using Microsoft.Extensions.FileProviders;

namespace CloneDatabase;

public class SqlDb
{
    private readonly string _connectionString;

    public SqlDb(string connectionString)
    {
        _connectionString = connectionString;
    }

    public List<DatabaseInfo> QueryDatabases()
    {
        return Query<DatabaseInfo>(
            "use [master];SELECT name, database_id as Id, create_date as CreateDate FROM sys.databases").ToList();
    }

    public List<string> QueryTableNames(string dbName)
    {
        return Query<string>(@$"use [{dbName}];
            SELECT info.TABLE_NAME as Name
            FROM {dbName}.INFORMATION_SCHEMA.TABLES as info WITH(NOLOCK)
            WHERE TABLE_TYPE = 'BASE TABLE';").ToList();
    }

    public TableInfo? QueryTable(string dbName, string tableName)
    {
        var fields = QueryFields(dbName, tableName);
        if (!fields.Any())
        {
            return null;
        }

        return new TableInfo
        {
            Name = tableName,
            Fields = fields
        };
    }

    public List<TableFieldInfo> QueryFields(string dbName, string tableName)
    {
        var sql = $@"use [{dbName}];
SELECT 
    c.name as Name,
    t.Name as DataType,
    c.max_length as MaxLength,
    c.precision ,
    c.scale,
    c.is_nullable as IsNullable,
    ISNULL(i.is_primary_key, 0) as IsPrimaryKey
FROM    
    sys.columns c
INNER JOIN 
    sys.types t ON c.user_type_id = t.user_type_id
LEFT OUTER JOIN 
    sys.index_columns ic ON ic.object_id = c.object_id AND ic.column_id = c.column_id
LEFT OUTER JOIN 
    sys.indexes i ON ic.object_id = i.object_id AND ic.index_id = i.index_id
WHERE
    c.object_id = OBJECT_ID('{tableName}')";
        return Query<TableFieldInfo>(sql).ToList();
    }

    public List<TableIndexInfo> QueryTableIndexList(string dbName, string tableName)
    {
        var sql = $@"use [{dbName}];
SELECT
 a.name AS Name,
 OBJECT_NAME(a.object_id) as TableName,
 COL_NAME(b.object_id, b.column_id) AS ColumnName,
 b.index_column_id as IndexColumnId,
 b.key_ordinal as KeyOrdinal,
 b.is_included_column as IsIncludedColumn
FROM sys.indexes AS a WITH(NOLOCK)
INNER JOIN sys.index_columns AS b WITH(NOLOCK) ON a.object_id = b.object_id AND a.index_id = b.index_id
WHERE
a.is_hypothetical = 0 AND
a.object_id = OBJECT_ID('dbo.[{tableName}]')
";
        return Query<TableIndexInfo>(sql).ToList();
    }

    public List<TableConstraintInfo> QueryConstraints(string dbName)
    {
        var sql = $@"use [{dbName}];
SELECT table_view as TableViewName,
    object_type as ObjectType, 
    constraint_type as ConstraintType,
    constraint_name as Name,
    details
FROM (
    SELECT schema_name(t.schema_id) + '.' + t.[name] as table_view, 
        case when t.[type] = 'U' then 'Table'
            when t.[type] = 'V' then 'View'
            end as [object_type],
        case when c.[type] = 'PK' then 'Primary key'
            when c.[type] = 'UQ' then 'Unique constraint'
            when i.[type] = 1 then 'Unique clustered index'
            when i.type = 2 then 'Unique index'
            end as constraint_type, 
        isnull(c.[name], i.[name]) as constraint_name,
        substring(column_names, 1, len(column_names)-1) as [details]
    FROM sys.objects t
        left outer join sys.indexes i
            on t.object_id = i.object_id
        left outer join sys.key_constraints c
            on i.object_id = c.parent_object_id 
            AND i.index_id = c.unique_index_id
       cross apply (select col.[name] + ', '
                        from sys.index_columns ic
                            inner join sys.columns col
                                on ic.object_id = col.object_id
                                and ic.column_id = col.column_id
                        where ic.object_id = t.object_id
                            AND ic.index_id = i.index_id
                                order by col.column_id
                                for xml path ('') ) D (column_names)
    WHERE is_unique = 1
        AND t.is_ms_shipped <> 1
    UNION ALL
    SELECT schema_name(fk_tab.schema_id) + '.' + fk_tab.name as foreign_table,
        'Table',
        'Foreign key',
        fk.name as fk_constraint_name,
        schema_name(pk_tab.schema_id) + '.' + pk_tab.name
    FROM sys.foreign_keys fk
        inner join sys.tables fk_tab
            on fk_tab.object_id = fk.parent_object_id
        inner join sys.tables pk_tab
            on pk_tab.object_id = fk.referenced_object_id
        inner join sys.foreign_key_columns fk_cols
            on fk_cols.constraint_object_id = fk.object_id
    UNION ALL
    SELECT schema_name(t.schema_id) + '.' + t.[name],
        'Table',
        'Check constraint',
        con.[name] as constraint_name,
        con.[definition]
    FROM sys.check_constraints con
        left outer join sys.objects t
            on con.parent_object_id = t.object_id
        left outer join sys.all_columns col
            on con.parent_column_id = col.column_id
            AND con.parent_object_id = col.object_id
    UNION ALL
    SELECT schema_name(t.schema_id) + '.' + t.[name],
        'Table',
        'Default constraint',
        con.[name],
        col.[name] + ' = ' + con.[definition]
    FROM sys.default_constraints con
        left outer join sys.objects t
            on con.parent_object_id = t.object_id
        left outer join sys.all_columns col
            on con.parent_column_id = col.column_id
            and con.parent_object_id = col.object_id ) a
ORDER BY table_view, constraint_type, constraint_name;";
        return Query<TableConstraintInfo>(sql).ToList();
    }

    public void QueryDepends()
    {
        var sql = @"SELECT referencing_schema_name, referencing_entity_name,
 referencing_id, referencing_class_desc, is_caller_dependent
 FROM sys.dm_sql_referencing_entities ('dbo.SimpleSettings', 'OBJECT')";
    }

    public List<T> Query<T>(string sql, object? dbParameter = null)
    {
        using var conn = new SqlConnection(_connectionString);
        return conn.Query<T>(sql, dbParameter).ToList();
    }

    public int Execute(string sql, object? param = null)
    {
        using var conn = new SqlConnection(_connectionString);
        return conn.Execute(sql, param);
    }
}

public static class QueryDatabaseMetaExtension
{
    public static void QueryPrimaryKeyConstraint(this IEnumerable<TableConstraintInfo> list)
    {
        foreach (var info in list)
        {   
            //if( info.ConstraintType == )
        }
    }
}