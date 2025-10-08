using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;
using T1.Standard.IO;

namespace CodeBoyLib.Services;

public class DatabaseDtoGenerator
{
    private string NormalName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return name;

        var result = name;

        if (result.Contains("."))
        {
            var parts = result.Split('.');
            result = parts[^1];
        }

        result = result.Replace("[", "").Replace("]", "");

        return result;
    }

    public string GenerateEfDtoCode(string createTableSql)
    {
        var sqlParser = new SqlParser(createTableSql);
        var result = sqlParser.Extract().ToArray();
        var output = new IndentStringBuilder();
        foreach (var sqlExpression in result)
        {
            var createTableExpr = sqlExpression as SqlCreateTableExpression;
            if (createTableExpr == null)
            {
                continue;
            }
            var tableName = NormalName(createTableExpr.TableName);
            output.WriteLine($"public class {tableName}Dto {{");
            output.Indent++;
            foreach (var column in createTableExpr.Columns)
            {
                if (column is SqlColumnDefinition columnDefinition)
                {
                    var columnName = NormalName(columnDefinition.ColumnName);
                    var dataType = SqlTypeToCsharpType(columnDefinition.DataType, columnDefinition.DataSize);
                    output.WriteLine($"public {dataType} {columnName} {{ get; set; }}");
                    continue;
                }

                if (column is SqlComputedColumnDefinition columnComputed)
                {
                    //var columnName = columnComputed.ColumnName;
                    //output.WriteLine($"public {dataType} {columnName} {{ get; set; }}");
                    continue;
                }
            }
            output.Indent--;
            output.WriteLine("}");
        }
        return output.ToString();
    }

    private string SqlTypeToCsharpType(string sqlType, SqlDataSize? sqlDataSize)
    {
        var normalizedType = sqlType.Replace("[", "").Replace("]", "");
        var upperSqlType = normalizedType.ToUpper();
        
        return upperSqlType switch
        {
            "INT" or "INTEGER" => "int",
            "BIGINT" => "long",
            "SMALLINT" => "short",
            "TINYINT" => "byte",
            "BIT" => "bool",
            "DECIMAL" or "NUMERIC" or "MONEY" or "SMALLMONEY" => "decimal",
            "FLOAT" or "REAL" => "double",
            "DATE" => "DateTime",
            "DATETIME" or "DATETIME2" or "SMALLDATETIME" => "DateTime",
            "TIME" => "TimeSpan",
            "DATETIMEOFFSET" => "DateTimeOffset",
            "CHAR" or "VARCHAR" or "NCHAR" or "NVARCHAR" or "TEXT" or "NTEXT" => "string",
            "BINARY" or "VARBINARY" or "IMAGE" => "byte[]",
            "UNIQUEIDENTIFIER" => "Guid",
            "XML" => "string",
            "JSON" => "string",
            _ => "object"
        };
    }
}