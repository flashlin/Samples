namespace T1.SqlSharp.Expressions;

public class LinqExpr
{
    public LinqFromExpr From { get; set; }
    public LinqWhereExpr? Where { get; set; }
    public LinqOrderByExpr? OrderBy { get; set; }
    public List<LinqJoinExpr>? Joins { get; set; }
    public List<LinqFromExpr>? AdditionalFroms { get; set; }
    public ILinqExpression? Select { get; set; }
}

public class LinqFromExpr
{
    public string Source { get; set; }
    public string AliasName { get; set; }
    public bool IsDefaultIfEmpty { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqFromExpr other) return false;
        return Source == other.Source && AliasName == other.AliasName && IsDefaultIfEmpty == other.IsDefaultIfEmpty;
    }
    public override int GetHashCode() => (Source, AliasName, IsDefaultIfEmpty).GetHashCode();
}

public class LinqSelectAllExpr : ILinqExpression
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

public class LinqOrderByExpr : ILinqExpression
{
    public List<LinqOrderByFieldExpr> Fields { get; set; } = new();
    public override bool Equals(object? obj)
    {
        if (obj is not LinqOrderByExpr other) return false;
        return Fields.SequenceEqual(other.Fields);
    }
    public override int GetHashCode() => Fields.GetHashCode();
}

public class LinqOrderByFieldExpr : ILinqExpression
{
    public LinqFieldExpr Field { get; set; }
    public bool IsDescending { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqOrderByFieldExpr other) return false;
        return Equals(Field, other.Field) && IsDescending == other.IsDescending;
    }
    public override int GetHashCode() => (Field, IsDescending).GetHashCode();
}

public class LinqJoinExpr : ILinqExpression
{
    public string JoinType { get; set; } = "join";
    public string AliasName { get; set; }
    public string Source { get; set; }
    public LinqConditionExpression On { get; set; }
    public string? Into { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqJoinExpr other) return false;
        return JoinType == other.JoinType
            && AliasName == other.AliasName
            && Source == other.Source
            && Equals(On, other.On)
            && Into == other.Into;
    }
    public override int GetHashCode() => (JoinType, AliasName, Source, On, Into).GetHashCode();
}

public class LinqSelectNewExpr : ILinqExpression
{
    public List<LinqSelectFieldExpr> Fields { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqSelectNewExpr other) return false;
        return Fields.SequenceEqual(other.Fields);
    }
    public override int GetHashCode() => Fields.GetHashCode();
}

public class LinqSelectFieldExpr : ILinqExpression
{
    public string Name { get; set; }
    public ILinqExpression Value { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqSelectFieldExpr other) return false;
        return Name == other.Name && Equals(Value, other.Value);
    }
    public override int GetHashCode() => (Name, Value).GetHashCode();
}