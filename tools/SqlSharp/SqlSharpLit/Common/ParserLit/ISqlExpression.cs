namespace SqlSharpLit.Common.ParserLit;

public enum SqlType
{
    None,
    Collection,
    CreateTable,
    Select,
    ColumnDefinition,
    Field,
    IntValue,
    ParameterValue,
    AddExtendedProperty,
    Constraint,
    String,
    Identity,
    WithToggle,
    Token
}

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}

public class SqlToken : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Token;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return Value;
    }
}