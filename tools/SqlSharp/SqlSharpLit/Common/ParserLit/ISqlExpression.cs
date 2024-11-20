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
    ConstraintUnique,
    ConstraintDefault
}

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}