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
        var lexer = new TSQLLexer(inputStream1);
        var tokenStream = new CommonTokenStream(lexer);
        var parser = new TSQLParser(tokenStream);

        // 解析第一個查詢語句
        IParseTree tree1 = parser.start();

        // 輸出解析樹
        Console.WriteLine(tree1.ToStringTree(parser));
    }

    public SqlExpr Parse(string sql)
    {
        var inputStream1 = new AntlrInputStream(sql);
        var lexer = new TSQLLexer(inputStream1);
        var tokenStream = new CommonTokenStream(lexer);
        var parser = new TSQLParser(tokenStream);
        var tree = parser.start();
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

public class SqlExprVisitor : TSQLBaseVisitor<SqlExpr>
{
    public override SqlExpr VisitSelectStatement(TSQLParser.SelectStatementContext context)
    {
        var exprList = new List<SqlExpr>();
        foreach (var columnContext in context.selectColumnList().selectColumn())
        {
            exprList.Add(Visit(columnContext));
        }

        var fromExpr = SqlExpr.Empty;
        var fromContext = context.fromClause();
        if (fromContext != null)
        {
            fromExpr = Visit(fromContext);
        }

        return new SelectExpr
        {
            Columns = exprList,
            FromClause = fromExpr
        };
    }

    public override SqlExpr VisitSelectColumn(TSQLParser.SelectColumnContext context)
    {
        var name = context.ID(0).GetText()!;
        var aliasName = context.ID(1)?.GetText() ?? string.Empty;
        return new FieldExpr
        {
            Name = name,
            AliasName = aliasName
        };
    }

    public override SqlExpr VisitTableReference(TSQLParser.TableReferenceContext context)
    {
        var tableName = context.ID(0).GetText()!;
        var aliasName = context.ID(1)?.GetText() ?? string.Empty;
        return new TableExpr
        {
            Name = tableName,
            AliasName = aliasName
        };
    }

    public override SqlExpr VisitFromClause(TSQLParser.FromClauseContext context)
    {
        var tableRefContext = context.tableReference();
        if (tableRefContext != null)
        {
            return VisitTableReference(tableRefContext);
        }

        var selectStatementContext = context.selectStatement();
        if (selectStatementContext != null)
        {
            return VisitSelectStatement(selectStatementContext);
        }

        return null;
    }
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