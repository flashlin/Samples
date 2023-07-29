using T1.ParserKit.ExprCollection;

namespace T1.ParserKit;

public class SqlExprVisitor : TSqlParserBaseVisitor<SqlExpr>
{
    public override SqlExpr VisitQuery_specification(TSqlParser.Query_specificationContext context)
    {
        var exprList = new List<SqlExpr>();
        foreach (var columnContext in context.columns.select_list_elem())
        {
            var sqlExpr = Visit(columnContext);
            exprList.Add(sqlExpr);
        }

        var fromExpr = SqlExpr.Empty;
        if (context.from != null)
        {
            fromExpr = Visit(context.from);
        }

        return new SelectExpr
        {
            Columns = exprList,
            FromClause = fromExpr
        };
    }

    public override SqlExpr VisitExpression_elem(TSqlParser.Expression_elemContext context)
    {
        //var leftAlias= context.leftAlias;
        if (context.expressionAs != null)
        {
            var name = context.expression().GetText()!;

            var aliasName = string.Empty;
            if (context.as_column_alias() != null)
            {
                var aliasExpr = Visit(context.as_column_alias()) as AliasExpr;
                aliasName = aliasExpr?.Name ?? string.Empty;
            }

            return new FieldExpr
            {
                Name = name,
                AliasName = aliasName
            };
        }
        
        return base.VisitExpression_elem(context);
    }

    public override SqlExpr VisitAs_column_alias(TSqlParser.As_column_aliasContext context)
    {
        //var _ = context.AS();
        var aliasName = context.column_alias();
        if (aliasName != null)
        {
            return new AliasExpr
            {
                Name = aliasName.GetText()
            };
        }
        return base.VisitAs_column_alias(context);
    }

    public override SqlExpr VisitTable_source_item(TSqlParser.Table_source_itemContext context)
    {
        var name = context.full_table_name().GetText()!;
        var aliasName = context.as_table_alias()?.GetText() ?? string.Empty;
        return new TableExpr
        {
            Name = name,
            AliasName = aliasName
        };
    }
}