using Antlr4.Runtime;
using Antlr4.Runtime.Tree;
using T1.ParserKit.ExprCollection;

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
        var inputStream = new AntlrInputStream(sql);
        var lexer = new TSqlLexer(inputStream);
        var tokenStream = new CommonTokenStream(lexer);
        var parser = new TSqlParser(tokenStream);
        var tree = parser.sql_clauses();
        var visitor = new SqlExprVisitor();
        var expr = visitor.Visit(tree);
        return expr;
    }
}