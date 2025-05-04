namespace T1.SqlSharp.Expressions;

public class LinqExpr
{
    public LinqFromExpr From { get; set; }
    public LinqSelectAllExpr Select { get; set; }
    public override bool Equals(object? obj)
    {
        if (obj is not LinqExpr other) return false;
        return Equals(From, other.From) && Equals(Select, other.Select);
    }
    public override int GetHashCode() => (From, Select).GetHashCode();
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