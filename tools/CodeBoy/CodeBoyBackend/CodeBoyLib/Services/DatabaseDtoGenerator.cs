using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;
using T1.Standard.IO;

namespace CodeBoyLib.Services;

public class DatabaseDtoGenerator
{
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
            var tableName = createTableExpr.TableName;
            output.WriteLine($"public class {tableName}Dto {{");
            output.Indent++;
            foreach (var column in createTableExpr.Columns)
            {
                if (column is SqlColumnDefinition columnDefinition)
                {
                    var columnName = columnDefinition.ColumnName;
                    var dataType = columnDefinition.DataType;
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
}