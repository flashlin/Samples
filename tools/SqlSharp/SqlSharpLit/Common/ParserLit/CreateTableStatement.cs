namespace SqlSharpLit.Common.ParserLit;

public class CreateTableStatement : ISqlExpression
{
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDefinition> Columns { get; set; } = [];
    public List<SqlConstraint> Constraints { get; set; } = [];

    public string ToSql()
    {
        var sql = new T1.Standard.IO.IndentStringBuilder();
        sql.WriteLine($"CREATE TABLE {TableName}");
        return sql.ToString();
    }
}