using FluentAssertions;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlVisitorTest
{
    [Test]
    public void Visit_cross_join_should_not_throw_when_on_condition_is_null()
    {
        var statement = "select id from customer c cross join emp e".ParseSql().ResultValue;

        var act = () => new SqlVisitor().Visit(statement);

        act.Should().NotThrow();
    }

    [Test]
    public void Visit_cross_apply_should_not_throw_when_on_condition_is_null()
    {
        var statement = "select id from customer c cross apply fn_items(c.id) t".ParseSql().ResultValue;

        var act = () => new SqlVisitor().Visit(statement);

        act.Should().NotThrow();
    }

    [Test]
    public void Visit_cross_join_should_collect_join_condition_node()
    {
        var statement = "select id from customer c cross join emp e".ParseSql().ResultValue;

        var nodes = new SqlVisitor().Visit(statement);

        nodes.Select(node => node.Expression.SqlType)
            .Should().Contain(SqlType.JoinCondition);
    }
}
