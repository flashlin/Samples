using System.Collections.Immutable;

namespace PidginDemo.LinqExpressions;

public class CallExpr : LinqExpr
{
    public LinqExpr Expr { get; }

    public ImmutableArray<LinqExpr> Arguments { get; }

    public CallExpr(LinqExpr expr, ImmutableArray<LinqExpr> arguments)
    {
        Expr = expr;
        Arguments = arguments;
    }

    public override bool Equals(LinqExpr? other)
        => other is CallExpr x
           && Expr.Equals(x.Expr)
           && Arguments.SequenceEqual(x.Arguments);

    public override int GetHashCode() => HashCode.Combine(Expr, Arguments);
}