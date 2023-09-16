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
    public string AliasTable { get; init; }
    public TableExpr Source { get; init; }
    
    public override bool Equals(LinqExpr? other)
        => other is SelectExpr x
           && AliasTable.Equals(x.AliasTable)
           && Source.Equals(x.Source);

    public override int GetHashCode() => HashCode.Combine(AliasTable, Source);
}

public class TableExpr : LinqExpr
{
    public string Database { get; init; } = string.Empty;
    public string Schema { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    
    public override bool Equals(LinqExpr? other)
        => other is TableExpr x
           && Database.Equals(x.Database)
           && Schema.Equals(x.Schema)
           && Name.Equals(x.Name)
           ;

    public override int GetHashCode() => HashCode.Combine(Database, Schema, Name);
}
