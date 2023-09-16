namespace PidginDemo.LinqExpressions;

public class UnaryOpExpr : LinqExpr
{
    public UnaryOperatorType Type { get; }

    public LinqExpr Expr { get; }

    public UnaryOpExpr(UnaryOperatorType type, LinqExpr expr)
    {
        Type = type;
        Expr = expr;
    }

    public override bool Equals(LinqExpr? other)
        => other is UnaryOpExpr u
           && Type == u.Type
           && Expr.Equals(u.Expr);

    public override int GetHashCode() => HashCode.Combine(Type, Expr);
}

public class SelectExpr : LinqExpr
{
    public string AliasTable { get; set; }
    
    public override bool Equals(LinqExpr? other)
        => other is SelectExpr x
           && AliasTable.Equals(x.AliasTable);

    public override int GetHashCode() => HashCode.Combine(AliasTable);
}