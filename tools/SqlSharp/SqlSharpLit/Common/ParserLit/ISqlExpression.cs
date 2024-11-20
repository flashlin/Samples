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
    Constraint,
    ConstraintDefault
}

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}