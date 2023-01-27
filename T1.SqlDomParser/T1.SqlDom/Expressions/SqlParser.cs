using System.Text;
using System.Text.RegularExpressions;
using Antlr4.Runtime;
using Antlr4.Runtime.Tree;
using T1.SqlDom.Tsql;
using T1.SqlDomParser;

namespace T1.SqlDom.Expressions;

public class SqlParser
{
    public SqlExpr Parse(string input)
    {
        var stream = new AntlrInputStream(input);
        var lexer = new TsqlLexer(stream); // 這個 HelloLexer 是透過 ANTLR 產生的
        var tokens = new CommonTokenStream(lexer);
        var parser = new TsqlParser(tokens)
        {
            BuildParseTree = true
        };
        var compileUnit = parser.select_statement()!;

        var visitor = new TsqlExprVisitor();
        var expr = visitor.Visit(compileUnit);
        return expr;

        var listener = new TsqlFormatListener();
        var walker = new ParseTreeWalker();
        walker.Walk(listener, compileUnit);
        var sql = listener.GetFormattedSql();
        throw new InvalidOperationException();
    }
}

public class NumberSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; init; } = null!;
}

public class TsqlFormatListener : TsqlParserBaseListener
{
    StringBuilder _formattedSql = new StringBuilder();

    public string GetFormattedSql()
    {
        return _formattedSql.ToString();
    }

    public override void EnterSelect_statement(TsqlParser.Select_statementContext context)
    {
        _formattedSql.AppendLine("SELECT ");
        base.EnterSelect_statement(context);
    }

    public override void ExitSelect_list(TsqlParser.Select_listContext context)
    {
        foreach (var item in context.column_elem())
        {
            var text = item.GetText();
            _formattedSql.AppendLine(text);
        }

        base.ExitSelect_list(context);
    }
}