using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlConditionExpression : ISqlWhereExpression 
{
    public required ISqlExpression Left { get; set; }
    public string Operation { get; set; } = string.Empty;
    public LogicalOperator LogicalOperator { get; set; } = LogicalOperator.None;

    public required ISqlExpression Right { get; set; }
    public string ToSql()
    {
        return $"{Left.ToSql()} {Operation} {Right.ToSql()}";
    }
}

public enum JoinType
{
    Inner,
    Left,
    Right,
    Full
}

public enum GroupingType
{
    Simple,
    GroupingSets,
    Cube,
    Rollup
}

public enum ComparisonOperator
{
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Like,
    In,
    Between,
    IsNull,
    IsNotNull
}

public enum LogicalOperator
{
    None,
    And,
    Or,
    Not
}

public class TableSource
{
    public string TableName { get; set; }

    public string Alias { get; set; }

    public JoinCondition Join { get; set; }
}

public class JoinCondition
{
    public JoinType Type { get; set; }

    public TableSource JoinedTable { get; set; }

    public string OnCondition { get; set; }
}