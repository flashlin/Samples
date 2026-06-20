using FluentAssertions;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlVisitorTest
{
    private static List<SqlType> VisitTypes(string sql)
    {
        var statement = sql.ParseSql().ResultValue;
        return new SqlVisitor()
            .Visit(statement)
            .Select(node => node.Expression.SqlType)
            .ToList();
    }

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
        var types = VisitTypes("select id from customer c cross join emp e");

        types.Should().Contain(SqlType.JoinCondition);
    }

    [Test]
    public void Visit_simple_select_should_collect_statement_column_and_table()
    {
        var types = VisitTypes("select id from customer");

        types.Should().Contain(SqlType.SelectStatement);
        types.Should().Contain(SqlType.SelectColumn);
        types.Should().Contain(SqlType.TableSource);
    }

    [Test]
    public void Visit_select_with_where_should_collect_condition()
    {
        var types = VisitTypes("select id from customer where id = 1");

        types.Should().Contain(SqlType.ComparisonCondition);
    }

    [Test]
    public void Visit_inner_join_should_collect_join_and_both_tables()
    {
        var types = VisitTypes("select id from customer c inner join emp e on c.id = e.id");

        types.Should().Contain(SqlType.JoinCondition);
        types.Count(type => type == SqlType.TableSource).Should().Be(2);
    }

    [Test]
    public void Visit_group_by_having_should_collect_clauses_and_function()
    {
        var types = VisitTypes("select id from customer group by id having count(1) = 2");

        types.Should().Contain(SqlType.GroupByClause);
        types.Should().Contain(SqlType.HavingClause);
        types.Should().Contain(SqlType.Function);
    }

    [Test]
    public void Visit_order_by_should_collect_clause_and_column()
    {
        var types = VisitTypes("select id from customer order by id");

        types.Should().Contain(SqlType.OrderByClause);
        types.Should().Contain(SqlType.OrderColumn);
    }

    [Test]
    public void Visit_union_should_collect_union_and_two_statements()
    {
        var types = VisitTypes("select id from a union select id from b");

        types.Should().Contain(SqlType.UnionSelect);
        types.Count(type => type == SqlType.SelectStatement).Should().BeGreaterThanOrEqualTo(2);
    }

    [Test]
    public void Visit_cte_should_collect_with_cte_and_common_table_expression()
    {
        var types = VisitTypes("with cte as (select id from customer) select id from cte");

        types.Should().Contain(SqlType.WithCte);
        types.Should().Contain(SqlType.CommonTableExpression);
    }

    [Test]
    public void Visit_case_when_should_collect_case_and_when_then()
    {
        var types = VisitTypes("select case when id = 1 then 'a' else 'b' end from customer");

        types.Should().Contain(SqlType.CaseClause);
        types.Should().Contain(SqlType.WhenThen);
    }

    [Test]
    public void Visit_between_should_collect_between_value()
    {
        var types = VisitTypes("select id from customer where id between 1 and 10");

        types.Should().Contain(SqlType.BetweenValue);
    }

    [Test]
    public void Visit_exists_should_collect_exists_expression()
    {
        var types = VisitTypes("select id from customer where exists (select 1 from emp)");

        types.Should().Contain(SqlType.ExistsExpression);
    }

    [Test]
    public void Visit_top_should_collect_top_clause()
    {
        var types = VisitTypes("select top 10 id from customer");

        types.Should().Contain(SqlType.TopClause);
    }

    [Test]
    public void Visit_create_table_should_collect_table_and_column_definition()
    {
        var types = VisitTypes("create table t (id int)");

        types.Should().Contain(SqlType.CreateTable);
        types.Should().Contain(SqlType.ColumnDefinition);
    }
}
