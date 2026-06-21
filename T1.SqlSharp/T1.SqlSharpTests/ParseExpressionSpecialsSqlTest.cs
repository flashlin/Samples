using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseExpressionSpecialsSqlTest
{
    [Test]
    public void Odbc_escape_function()
    {
        var sql = "SELECT { fn UCASE(Name) } FROM Users";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var field = result!.Columns[0].Field as SqlOdbcEscapeExpr;
        Assert.That(field, Is.Not.Null);
        Assert.That(field!.Keyword, Is.EqualTo("fn"));
        Assert.That(field.ToSql(), Is.EqualTo("{ fn UCASE(Name) }"));
    }

    [Test]
    public void Partition_function_expression()
    {
        var sql = "SELECT $PARTITION.RangePF(OrderDate) AS p FROM Orders";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var fn = result!.Columns[0].Field as SqlFunctionExpression;
        Assert.That(fn, Is.Not.Null, $"actual type: {result.Columns[0].Field?.GetType().Name}");
        Assert.That(fn!.FunctionName, Is.EqualTo("$PARTITION.RangePF"));
    }

    [Test]
    public void Odbc_escape_timestamp()
    {
        var sql = "SELECT { ts '2021-01-01 10:00:00' } AS dt";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var field = result!.Columns[0].Field as SqlOdbcEscapeExpr;
        Assert.That(field, Is.Not.Null);
        Assert.That(field!.Keyword, Is.EqualTo("ts"));
        Assert.That(field.ToSql(), Is.EqualTo("{ ts '2021-01-01 10:00:00' }"));
    }
}
