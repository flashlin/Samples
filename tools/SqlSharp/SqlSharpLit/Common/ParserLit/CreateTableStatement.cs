namespace SqlSharpLit.Common.ParserLit;

public class CreateTableStatement : ISqlExpression
{
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDefinition> Columns { get; set; } = [];
    public List<SqlConstraint> Constraints { get; set; } = [];
}