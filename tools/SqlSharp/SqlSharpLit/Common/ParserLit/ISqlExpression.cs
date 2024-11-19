namespace SqlSharpLit.Common.ParserLit;

public enum SqlType
{
    Empty,
    CreateTable,
    Select,
    ColumnDefinition,
    Field,
    IntValue,
    ParameterValue,
    AddExtendedProperty,
    Constraint
}

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}