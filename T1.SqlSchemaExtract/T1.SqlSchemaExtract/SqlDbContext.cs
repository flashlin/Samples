using Dapper;
using Microsoft.Data.SqlClient;

namespace T1.SqlSchemaExtract;

public class SqlDbContext : IDisposable, IAsyncDisposable
{
    private SqlConnection? _connection;

    public async Task OpenAsync(string connectionString)
    {
        await CloseAsync();
        _connection = new SqlConnection(connectionString);
        await _connection.OpenAsync();
    }

    public async Task CloseAsync()
    {
        if (_connection == null)
        {
            return;
        }
        await _connection.CloseAsync();
    }

    public async Task<List<TableSchema>> QueryTableSchemaAsync()
    {
        var schemaInfoList = await QueryTableSchemaInfoAsync();
        return GroupTableSchemas(schemaInfoList);
    }

    public async Task<List<ForeignKey>> QueryForeignKeyAsync()
    {
        var foreignKeyInfoList = await QueryForeignKeyInfoAsync();
        return GroupForeignKeys(foreignKeyInfoList);
    }

    public List<TableSchema> GetTablesInDependencyOrder(List<TableSchema> tables, List<ForeignKey> foreignKeys)
    {
        // Build dependency graph
        var dependencies = new Dictionary<string, HashSet<string>>();
        var inDegree = new Dictionary<string, int>(); //被依賴的次數
        
        // Initialize all tables with zero dependencies
        foreach (var table in tables)
        {
            dependencies[table.Name] = new HashSet<string>();
            inDegree[table.Name] = 0;
        }
        
        // Build dependency relationships from foreign keys
        foreach (var fk in foreignKeys)
        {
            // Foreign table depends on primary table
            if (dependencies.ContainsKey(fk.ForeignTableName) && 
                inDegree.ContainsKey(fk.PrimaryTableName))
            {
                if (dependencies[fk.ForeignTableName].Add(fk.PrimaryTableName))
                {
                    inDegree[fk.ForeignTableName]++;
                }
            }
        }
        
        // Topological sort using Kahn's algorithm
        var queue = new Queue<string>();
        var result = new List<TableSchema>();
        // Start with tables that have no dependencies
        foreach (var kvp in inDegree.Where(x => x.Value == 0))
        {
            queue.Enqueue(kvp.Key);
        }
        
        while (queue.Count > 0)
        {
            var currentTable = queue.Dequeue();
            var tableSchema = tables.FirstOrDefault(t => t.Name == currentTable);
            if (tableSchema != null)
            {
                result.Add(tableSchema);
            }
            
            // Process all tables that depend on current table
            foreach (var dependentTable in dependencies.Keys)
            {
                if (dependencies[dependentTable].Contains(currentTable))
                {
                    dependencies[dependentTable].Remove(currentTable);
                    inDegree[dependentTable]--;
                    
                    if (inDegree[dependentTable] == 0)
                    {
                        queue.Enqueue(dependentTable);
                    }
                }
            }
        }
        
        // Check for circular dependencies
        if (result.Count != tables.Count)
        {
            // Add remaining tables (those with circular dependencies) at the end
            var remainingTables = tables.Where(t => result.All(r => r.Name != t.Name));
            result.AddRange(remainingTables);
        }
        
        return result;
    }

    private static List<TableSchema> GroupTableSchemas(List<TableSchemaInfo> schemaInfoList)
    {
        return schemaInfoList
            .GroupBy(info => info.TableName)
            .Select(group => new TableSchema
            {
                Name = group.Key,
                Fields = group.Select(info => new FieldSchema
                {
                    Name = info.FieldName,
                    DataType = info.FieldDataType,
                    DataSize = info.FieldDataSize,
                    DataScale = info.FieldDataScale,
                    IsPrimaryKey = info.IsPrimaryKey,
                    IsNullable = info.IsNullable,
                    IsIdentity = info.IsIdentity,
                    DefaultValue = info.DefaultValue,
                    Description = info.Description
                }).ToList()
            })
            .ToList();
    }

    private static List<ForeignKey> GroupForeignKeys(List<ForeignKeyInfo> foreignKeyInfoList)
    {
        return foreignKeyInfoList
            .GroupBy(info => info.DefineName)
            .Select(group => new ForeignKey
            {
                DefineName = group.Key,
                ForeignTableName = group.First().ForeignTableName,
                PrimaryTableName = group.First().PrimaryTableName,
                ForeignKeyNames = group.Select(info => info.ForeignKeyName).ToList(),
                PrimaryKeyNames = group.Select(info => info.PrimaryKeyName).ToList()
            })
            .ToList();
    }

    private async Task<List<TableSchemaInfo>> QueryTableSchemaInfoAsync()
    {
        var sql = """
                 SELECT
                     t.name AS TableName,
                     c.name AS FieldName,
                     ty.name AS FieldDataType,
                     c.max_length AS FieldDataSize,
                     c.scale AS FieldDataScale,
                     CASE 
                         WHEN pk.column_id IS NOT NULL THEN 1
                         ELSE 0
                     END AS IsPrimaryKey,
                     c.is_nullable AS IsNullable,
                     c.is_identity AS IsIdentity,
                     ISNULL(def.definition, '') AS DefaultValue,
                     ISNULL(p.value, '') AS Description
                 FROM
                     sys.tables AS t
                 INNER JOIN
                     sys.columns AS c ON t.object_id = c.object_id
                 INNER JOIN
                     sys.types AS ty ON c.user_type_id = ty.user_type_id
                 LEFT JOIN
                     sys.default_constraints AS def ON c.default_object_id = def.object_id
                 LEFT JOIN
                     sys.extended_properties AS p ON p.major_id = t.object_id AND p.minor_id = c.column_id AND p.name = 'MS_Description'
                 LEFT JOIN
                     (
                         SELECT
                             ic.object_id,
                             ic.column_id
                         FROM
                             sys.indexes AS i
                         INNER JOIN
                             sys.index_columns AS ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                         WHERE
                             i.is_primary_key = 1
                     ) AS pk ON pk.object_id = t.object_id AND pk.column_id = c.column_id
                 WHERE
                     t.is_ms_shipped = 0 -- 排除系統內建的 Table
                 ORDER BY 
                     t.Name
                 """;
        var q = await _connection!.QueryAsync<TableSchemaInfo>(sql);
        return q.ToList();
    }

    private async Task<List<ForeignKeyInfo>> QueryForeignKeyInfoAsync()
    {
        var sql = """
                 SELECT
                     fk.name AS DefineName,
                     ft.name AS ForeignTableName,
                     fc.name AS ForeignKeyName,
                     pt.name AS PrimaryTableName,
                     pc.name AS PrimaryKeyName
                 FROM
                     sys.foreign_keys AS fk
                 INNER JOIN
                     sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
                 INNER JOIN
                     sys.tables AS ft ON fk.parent_object_id = ft.object_id
                 INNER JOIN
                     sys.columns AS fc ON fkc.parent_object_id = fc.object_id AND fkc.parent_column_id = fc.column_id
                 INNER JOIN
                     sys.tables AS pt ON fk.referenced_object_id = pt.object_id
                 INNER JOIN
                     sys.columns AS pc ON fkc.referenced_object_id = pc.object_id AND fkc.referenced_column_id = pc.column_id
                 WHERE
                     ft.is_ms_shipped = 0 -- 排除系統內建的 Table
                     AND pt.is_ms_shipped = 0 -- 排除系統內建的 Table
                 ORDER BY
                     fk.name,
                     fkc.constraint_column_id;
                 """;
        var q = await _connection!.QueryAsync<ForeignKeyInfo>(sql);
        return q.ToList();
    }

    public async Task<List<NonClusteredIndexInfo>> QueryNonClusteredIndexesAsync()
    {
        var sql = """
                  SELECT
                      t.name AS TableName,
                      i.name AS IndexName,
                      c.name AS FieldName,
                      ic.is_descending_key AS IsDesc
                  FROM
                      sys.tables AS t
                  INNER JOIN
                      sys.indexes AS i ON t.object_id = i.object_id
                  INNER JOIN
                      sys.index_columns AS ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                  INNER JOIN
                      sys.columns AS c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                  WHERE
                      t.is_ms_shipped = 0 -- 排除系統內建的 Table
                      AND i.is_hypothetical = 0 -- 排除假設的索引
                      AND i.type = 2 -- 只取非叢集索引 (1=叢集, 2=非叢集)
                  ORDER BY
                      t.name,
                      i.name,
                      ic.key_ordinal;
                  """;
        var q = await _connection!.QueryAsync<TableIndexInfoRaw>(sql);
        var indexInfoList = q.ToList();
        return GroupTableIndexes(indexInfoList);
    }

    public async Task<List<ClusteredIndexInfo>> QueryClusteredIndexesAsync()
    {
        var sql = """
                  SELECT
                      t.name AS TableName,
                      i.name AS IndexName,
                      c.name AS FieldName,
                      ic.is_descending_key AS IsDesc
                  FROM
                      sys.tables AS t
                  INNER JOIN
                      sys.indexes AS i ON t.object_id = i.object_id
                  INNER JOIN
                      sys.index_columns AS ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                  INNER JOIN
                      sys.columns AS c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                  WHERE
                      t.is_ms_shipped = 0 -- 排除系統內建的 Table
                      AND i.is_hypothetical = 0 -- 排除假設的索引
                      AND i.type = 1 -- 只取叢集索引 (1=叢集, 2=非叢集)
                  ORDER BY
                      t.name,
                      i.name,
                      ic.key_ordinal;
                  """;
        var q = await _connection!.QueryAsync<TableIndexInfoRaw>(sql);
        var indexInfoList = q.ToList();
        return GroupClusteredIndexes(indexInfoList);
    }

    public async Task<List<TableConstraintInfo>> QueryTableConstraintsAsync()
    {
        var sql = """
                  SELECT
                      t.name AS TableName,
                      cc.name AS ConstraintName,
                      c.name AS FieldName
                  FROM
                      sys.tables AS t
                  INNER JOIN
                      sys.check_constraints AS cc ON t.object_id = cc.parent_object_id
                  INNER JOIN
                      sys.columns AS c ON cc.parent_object_id = c.object_id
                  WHERE
                      t.is_ms_shipped = 0 -- 排除系統內建的 Table
                  ORDER BY
                      t.name,
                      cc.name,
                      c.column_id;
                  """;
        var q = await _connection!.QueryAsync<TableConstraintInfoRaw>(sql);
        var constraintInfoList = q.ToList();
        return GroupTableConstraints(constraintInfoList);
    }

    private static List<NonClusteredIndexInfo> GroupTableIndexes(List<TableIndexInfoRaw> indexInfoList)
    {
        return indexInfoList
            .GroupBy(info => new { info.TableName, info.IndexName })
            .Select(group => new NonClusteredIndexInfo
            {
                TableName = group.Key.TableName,
                IndexName = group.Key.IndexName,
                IndexFields = group.Select(info => new FieldIndexInfo
                {
                    FieldName = info.FieldName,
                    IsDesc = info.IsDesc
                }).ToList()
            })
            .ToList();
    }

    private static List<ClusteredIndexInfo> GroupClusteredIndexes(List<TableIndexInfoRaw> indexInfoList)
    {
        return indexInfoList
            .GroupBy(info => new { info.TableName, info.IndexName })
            .Select(group => new ClusteredIndexInfo
            {
                TableName = group.Key.TableName,
                IndexName = group.Key.IndexName,
                IndexFields = group.Select(info => new FieldIndexInfo
                {
                    FieldName = info.FieldName,
                    IsDesc = info.IsDesc
                }).ToList()
            })
            .ToList();
    }

    private static List<TableConstraintInfo> GroupTableConstraints(List<TableConstraintInfoRaw> constraintInfoList)
    {
        return constraintInfoList
            .GroupBy(info => new { info.TableName, info.ConstraintName })
            .Select(group => new TableConstraintInfo
            {
                TableName = group.Key.TableName,
                ConstraintName = group.Key.ConstraintName,
                FieldNames = group.Select(info => info.FieldName).ToList()
            })
            .ToList();
    }

    private async Task<List<TableIndexInfoRaw>> QueryTableIndexInfoAsync()
    {
        var sql = """
                 SELECT
                     t.name AS TableName,
                     i.name AS IndexName,
                     c.name AS FieldName,
                     ic.is_descending_key AS IsDesc
                 FROM
                     sys.tables AS t
                 INNER JOIN
                     sys.indexes AS i ON t.object_id = i.object_id
                 INNER JOIN
                     sys.index_columns AS ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                 INNER JOIN
                     sys.columns AS c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                 WHERE
                     t.is_ms_shipped = 0 -- 排除系統內建的 Table
                     AND i.is_hypothetical = 0 -- 排除假設的索引
                     AND i.type = 2 -- 只取非叢集索引 (1=叢集, 2=非叢集)
                 ORDER BY
                     t.name,
                     i.name,
                     ic.key_ordinal;
                 """;
        var q = await _connection!.QueryAsync<TableIndexInfoRaw>(sql);
        return q.ToList();
    }

    private class TableIndexInfoRaw
    {
        public string TableName { get; set; } = string.Empty;
        public string IndexName { get; set; } = string.Empty;
        public string FieldName { get; set; } = string.Empty;
        public bool IsDesc { get; set; }
    }

    private class TableConstraintInfoRaw
    {
        public string TableName { get; set; } = string.Empty;
        public string ConstraintName { get; set; } = string.Empty;
        public string FieldName { get; set; } = string.Empty;
    }

    public void Dispose()
    {
        _connection?.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        if (_connection != null)
        {
            await _connection.DisposeAsync();
        }
    }

    public static string BuildConnectionString(string server, string sa, string saPassword)
    {
        var serverParts = server.Split(':');
        var serverName = serverParts[0];
        var port = serverParts.Length > 1 ? $",{serverParts[1]}" : string.Empty;
        return $"Server={serverName}{port};User ID={sa};Password={saPassword};TrustServerCertificate=True;";
    }

    public async Task ExecuteAsync(string sql)
    {
        await _connection!.ExecuteAsync(sql);
    }
}