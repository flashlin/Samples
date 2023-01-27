using T1.SqlDom.Tsql;

namespace T1.SqlDom.Expressions;

public class TsqlExprVisitor : TsqlParserBaseVisitor<SqlExpr>
{
    public override SqlExpr VisitSelect_statement(TsqlParser.Select_statementContext context)
    {
        var selectExpr = new SelectExpr
        {
            Columns = Visit(context.select_list()).ToList()
        };
        return selectExpr;
    }

    public override SqlExpr VisitSelect_list(TsqlParser.Select_listContext context)
    {
        var expr = new SqlExprCollection();
        foreach (var column in context._selectElement)
        {
            expr.Items.Add(Visit(column));
        }
        return expr;
    }

    public override SqlExpr VisitConstant(TsqlParser.ConstantContext context)
    {
        var nDecimal = context.DECIMAL();
        if (nDecimal != null)
        {
            return new NumberSqlExpr()
            {
                Value = nDecimal.Symbol.Text
            };
        }

        var nInt = context.INT();
        if (nInt != null)
        {
            return new NumberSqlExpr
            {
                Value = nInt.Symbol.Text
            };
        }

        throw new NotSupportedException();
    }
}