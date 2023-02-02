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

    public override SqlExpr VisitColumn_elem(TsqlParser.Column_elemContext context)
    {
        var alias = string.Empty;
        var nAlias = context.as_column_alias();
        if (nAlias != null)
        {
            alias = nAlias.GetText();
        }

        var field = Visit(context.children[0]);

        return new ColumnSqlExpr
        {
            Field = field,
            Alias = alias,
        };
    }

    public override SqlExpr VisitFull_column_name(TsqlParser.Full_column_nameContext context)
    {
        return new TableSqlExpr
        {
            Name = Visit(context.children[0]),
        };
    }

    public override SqlExpr VisitId_(TsqlParser.Id_Context context)
    {
        return new IdSqlExpr
        {
            Value = context.GetText()
        };
    }

    public override SqlExpr VisitAs_column_alias(TsqlParser.As_column_aliasContext context)
    {
        return new AliasSqlExpr
        {
            Name = Visit(context.column_alias())
        };
    }

    public override SqlExpr VisitColumn_alias(TsqlParser.Column_aliasContext context)
    {
        var nId = context.id_();
        if (nId != null)
        {
            return new IdSqlExpr
            {
                Value = context.GetText()
            };
        }

        return new StringSqlExpr
        {
            Value = context.GetText()
        };
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

public class TableSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Name.ToSqlString();
    }

    public SqlExpr Name { get; init; } = SqlExpr.Empty;
}

public class StringSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; init; } = string.Empty;
}

public class IdSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; set; } = string.Empty;
}

public class AliasSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        throw new NotImplementedException();
    }

    public SqlExpr Name { get; set; } = SqlExpr.Empty;
}

public class ColumnSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        var alias = Alias == string.Empty ? string.Empty : " " + Alias;
        return $"{Field.ToSqlString()}{alias}";
    }

    public string Alias { get; init; } = string.Empty;
    public SqlExpr Field { get; init; } = SqlExpr.Empty;
}