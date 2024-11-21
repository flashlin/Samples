namespace SqlSharpLit.Common.ParserLit;

public class SqlFieldExpression : ISqlValue, ISqlExpression
{
    public SqlType SqlType => SqlType.Field;
    public string FieldName { get; set; } = string.Empty;
    public string ToSql()
    {
        return FieldName;
    }
    public string Value => FieldName;
}