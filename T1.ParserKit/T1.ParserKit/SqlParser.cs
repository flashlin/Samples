using Antlr4.Runtime;
using Antlr4.Runtime.Tree;

namespace T1.ParserKit;

public class SqlParser
{
    public void Test()
    {
        string query1 = "select id, name from table";

        // 建立輸入流
        var inputStream1 = new AntlrInputStream(query1);

        // 建立語法解析器
        var lexer = new TSqlLexer(inputStream1);
        var tokenStream = new CommonTokenStream(lexer);
        var parser = new TSqlParser(tokenStream);

        // 解析第一個查詢語句
        IParseTree tree1 = parser.sql_clauses();

        // 輸出解析樹
        Console.WriteLine(tree1.ToStringTree(parser));
    }

    public SqlExpr Parse(string sql)
    {
        var inputStream1 = new AntlrInputStream(sql);
        var lexer = new TSqlLexer(inputStream1);
        var tokenStream = new CommonTokenStream(lexer);
        var parser = new TSqlParser(tokenStream);
        var tree = parser.sql_clauses();
        var visitor = new SqlExprVisitor();
        var expr = visitor.Visit(tree);
        return expr;
    }
}

public class SelectExpr : SqlExpr
{
    public List<SqlExpr> Columns { get; set; } = new();
    public SqlExpr? FromClause { get; set; }
}

public class SourceExpr : SqlExpr
{
    public SqlExpr From { get; set; }
}

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

public class AliasExpr : SqlExpr
{
    public string Name { get; set; }
}

public class FieldExpr : SqlExpr
{
    public string Name { get; set; } = string.Empty;
    public string AliasName { get; set; } = string.Empty;
}

public class TableExpr : SqlExpr
{
    public string Name { get; set; } = string.Empty;
    public string AliasName { get; set; } = string.Empty;
}