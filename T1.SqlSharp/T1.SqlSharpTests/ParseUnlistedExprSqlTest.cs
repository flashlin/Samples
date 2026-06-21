using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseUnlistedExprSqlTest
{
    [Test]
    public void Select_variable_assignment()
    {
        var sql = "SELECT @x = Name FROM Users";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var assign = result!.Columns[0].Field as SqlAssignExpr;
        Assert.That(assign, Is.Not.Null);
        Assert.That(assign!.Left.ToSql(), Is.EqualTo("@x"));
        Assert.That(assign.Right.ToSql(), Is.EqualTo("Name"));
    }

    [Test]
    public void Is_distinct_from()
    {
        var sql = "SELECT * FROM t WHERE a IS DISTINCT FROM b";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var condition = result!.Where as SqlConditionExpression;
        Assert.That(condition, Is.Not.Null);
        Assert.That(condition!.ComparisonOperator, Is.EqualTo(ComparisonOperator.IsDistinctFrom));
    }

    [Test]
    public void Is_not_distinct_from()
    {
        var sql = "SELECT * FROM t WHERE a IS NOT DISTINCT FROM b";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var condition = result!.Where as SqlConditionExpression;
        Assert.That(condition, Is.Not.Null);
        Assert.That(condition!.ComparisonOperator, Is.EqualTo(ComparisonOperator.IsNotDistinctFrom));
    }

    [Test]
    public void Comparison_all_subquery()
    {
        var sql = "SELECT * FROM t WHERE x > ALL (SELECT y FROM s)";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var condition = result!.Where as SqlConditionExpression;
        Assert.That(condition, Is.Not.Null);
        var quantified = condition!.Right as SqlQuantifiedExpr;
        Assert.That(quantified, Is.Not.Null);
        Assert.That(quantified!.Quantifier, Is.EqualTo("ALL"));
        Assert.That(quantified.Subquery, Is.InstanceOf<SelectStatement>());
    }

    [Test]
    public void Comparison_any_subquery()
    {
        var sql = "SELECT * FROM t WHERE x <= ANY (SELECT y FROM s)";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var condition = result!.Where as SqlConditionExpression;
        var quantified = condition!.Right as SqlQuantifiedExpr;
        Assert.That(quantified, Is.Not.Null);
        Assert.That(quantified!.Quantifier, Is.EqualTo("ANY"));
    }

    [Test]
    public void Unicode_string_literal()
    {
        var sql = "SELECT N'hello world'";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var value = result!.Columns[0].Field as SqlValue;
        Assert.That(value, Is.Not.Null);
        Assert.That(value!.Value, Is.EqualTo("N'hello world'"));
    }

    [Test]
    public void Like_with_escape()
    {
        var sql = "SELECT * FROM t WHERE name LIKE '%50!%%' ESCAPE '!'";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var condition = result!.Where as SqlConditionExpression;
        Assert.That(condition, Is.Not.Null);
        Assert.That(condition!.ComparisonOperator, Is.EqualTo(ComparisonOperator.Like));
        Assert.That(condition.Escape, Is.EqualTo("'!'"));
    }
}
