namespace T1.SqlSharp.Expressions;

public class LinqExpr
{
    public LinqFromExpr From { get; set; }
    public LinqWhereExpr? Where { get; set; }
    public LinqSelectAllExpr Select { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqExpr other) return false;
        return Equals(From, other.From) && Equals(Where, other.Where) && Equals(Select, other.Select);
    }
    public override int GetHashCode() => (From, Where, Select).GetHashCode();
}

public class LinqFromExpr
{
    public string Source { get; set; }
    public string AliasName { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqFromExpr other) return false;
        return Source == other.Source && AliasName == other.AliasName;
    }
    public override int GetHashCode() => (Source, AliasName).GetHashCode();
}

public class LinqSelectAllExpr
{
    public string AliasName { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqSelectAllExpr other) return false;
        return AliasName == other.AliasName;
    }
    public override int GetHashCode() => AliasName.GetHashCode();
}

public class LinqWhereExpr
{
    public ILinqExpression Condition { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqWhereExpr other) return false;
        return Equals(Condition, other.Condition);
    }
    public override int GetHashCode() => Condition?.GetHashCode() ?? 0;
}

public interface ILinqExpression
{
    bool Equals(object? obj);
    int GetHashCode();
}

public class LinqConditionExpression : ILinqExpression
{
    public ILinqExpression Left { get; set; }
    public ComparisonOperator ComparisonOperator { get; set; }
    public ILinqExpression Right { get; set; }
    public LogicalOperator? LogicalOperator { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqConditionExpression other) return false;
        if (LogicalOperator.HasValue || other.LogicalOperator.HasValue)
        {
            return Equals(Left, other.Left) && Equals(Right, other.Right) && Nullable.Equals(LogicalOperator, other.LogicalOperator);
        }
        return Equals(Left, other.Left) && ComparisonOperator == other.ComparisonOperator && Equals(Right, other.Right);
    }
    public override int GetHashCode() => (Left, ComparisonOperator, Right, LogicalOperator).GetHashCode();
}

public class LinqFieldExpr : ILinqExpression
{
    public string? TableOrAlias { get; set; }
    public string FieldName { get; set; } = string.Empty;
    public override bool Equals(object? obj)
    {
        if (obj is not LinqFieldExpr other) return false;
        return TableOrAlias == other.TableOrAlias && FieldName == other.FieldName;
    }
    public override int GetHashCode() => (TableOrAlias, FieldName).GetHashCode();
}

public class LinqValue : ILinqExpression
{
    public string Value { get; set; } = string.Empty;
    public override bool Equals(object? obj)
    {
        if (obj is not LinqValue other) return false;
        return Value == other.Value;
    }
    public override int GetHashCode() => Value.GetHashCode();
}