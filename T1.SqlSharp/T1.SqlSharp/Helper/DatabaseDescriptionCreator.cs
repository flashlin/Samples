using T1.SqlSharp.DatabaseDescriptions;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharp.Helper;

public static class DatabaseDescriptionCreator
{
    public static DatabaseDescription CreateDatabaseDescription(string databaseName, List<ISqlExpression> sqlExpressions)
    {
        var database = new DatabaseDescription
        {
            DatabaseName = databaseName
        };
        var createTables = sqlExpressions.FilterCreateTableExpression();
        var sqlSpAddExtendedPropertyExpressions = sqlExpressions
            .FilterAddExtendedPropertyExpression();
        foreach (var createTable in createTables)
        {
            var table = CreateTableDescription(createTable, sqlSpAddExtendedPropertyExpressions);
            database.Tables.Add(table);
        }
        return database;
    }

    public static TableDescription CreateTableDescription(SqlCreateTableExpression createTable, 
        List<SqlSpAddExtendedPropertyExpression> sqlSpAddExtendedPropertyExpressions)
    {
        var tableName = createTable.TableName;
        var columns = createTable.Columns
            .Where(x => x.SqlType == SqlType.ColumnDefinition)
            .Cast<SqlColumnDefinition>()
            .ToList();
        var table = new TableDescription()
        {
            TableName = createTable.TableName,
            Columns = columns.Select(column =>
            {
                var columnDescription = CreateColumnDescription(column);
                columnDescription.Description = sqlSpAddExtendedPropertyExpressions
                    .GetColumnDescription(tableName, columnDescription.ColumnName);
                return columnDescription;
            }).ToList()
        };
        
        return table;
    }

    private static ColumnDescription CreateColumnDescription(SqlColumnDefinition column)
    {
        return new ColumnDescription()
        {
            ColumnName = column.ColumnName,
            DataType = CreateColumnDataType(column),
            IsNullable = column.IsNullable,
            IsIdentity = IsIdentity(column.Identity),
            DefaultValue = column.Constraints.Where(x => x.SqlType == SqlType.ConstraintDefaultValue)
                .Cast<SqlConstraintDefaultValue>()
                .Select(x => x.DefaultValue)
                .FirstOrDefault(string.Empty),
        };
    }

    private static string CreateColumnDataType(SqlColumnDefinition column)
    {
        if (column.DataSize != null)
        {
            return $"{column.DataType}{column.DataSize.ToSql()}";
        }
        return column.DataType;
    }

    private static bool IsIdentity(SqlIdentity sqlIdentity)
    {
        return sqlIdentity.Increment > 0;
    }
}