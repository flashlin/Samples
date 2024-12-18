namespace T1.SqlSharp.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
    TextSpan Span { get; set; }
    void Accept(SqlVisitor visitor);
}

public class SqlVisitor
{
    public virtual void Visit_SqlArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr sqlAliasExpr)
    {
    }

    public virtual void Visit_AsExpr(SqlAsExpr sqlAsExpr)
    {
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr sqlArithmeticBinaryExpr)
    {
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator sqlLogicalOperator)
    {
    }

    public virtual void Visit_SelectColumn(SelectColumn selectColumn)
    {
    }

    public virtual void Visit_SelectStatement(SelectStatement selectStatement)
    {
    }
}