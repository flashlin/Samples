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

    [Test]
    public void Visit_arithmetic_column_should_collect_arithmetic_binary_expr()
    {
        var types = VisitTypes("select id + 1 from customer");

        types.Should().Contain(SqlType.ArithmeticBinaryExpr);
    }

    [Test]
    public void Visit_cast_column_should_collect_as_expr()
    {
        var types = VisitTypes("select cast(id as nvarchar(3)) from customer");

        types.Should().Contain(SqlType.AsExpr);
    }

    [Test]
    public void Visit_unary_column_should_collect_unary_expr()
    {
        var types = VisitTypes("select ~id from customer");

        types.Should().Contain(SqlType.UnaryExpression);
    }

    [Test]
    public void Visit_parenthesized_case_should_collect_group_case_and_negative_value()
    {
        var types = VisitTypes("select (case when n = 0 and id > 0 then -id * 100 else 1 end) from customer");

        types.Should().Contain(SqlType.Group);
        types.Should().Contain(SqlType.CaseClause);
        types.Should().Contain(SqlType.WhenThen);
        types.Should().Contain(SqlType.NegativeValue);
    }

    [Test]
    public void Visit_not_exists_should_collect_not_expression()
    {
        var types = VisitTypes("select id from customer c where not exists (select 1 from emp)");

        types.Should().Contain(SqlType.NotExpression);
    }

    [Test]
    public void Visit_where_and_should_collect_search_condition()
    {
        var types = VisitTypes("select id from customer where id = 1 and name = 'a'");

        types.Should().Contain(SqlType.SearchCondition);
    }

    [Test]
    public void Visit_in_list_should_collect_values()
    {
        var types = VisitTypes("select id from customer where name in ('a','b')");

        types.Should().Contain(SqlType.Values);
    }

    [Test]
    public void Visit_over_order_by_should_collect_over_order_by()
    {
        var types = VisitTypes("select ROW_NUMBER() OVER (ORDER BY id DESC) as r from customer");

        types.Should().Contain(SqlType.OverOrderBy);
    }

    [Test]
    public void Visit_over_partition_by_should_collect_over_partition_by()
    {
        var types = VisitTypes("select sum(count(1)) over(partition by id) as Total from customer");

        types.Should().Contain(SqlType.OverPartitionByClause);
    }

    [Test]
    public void Visit_rank_should_collect_rank_partition_by_and_order_by()
    {
        var types = VisitTypes("select Rank() OVER (PARTITION BY id ORDER BY id) AS Row from customer");

        types.Should().Contain(SqlType.RankClause);
        types.Should().Contain(SqlType.PartitionBy);
        types.Should().Contain(SqlType.OrderByClause);
    }

    [Test]
    public void Visit_pivot_should_collect_pivot_clause()
    {
        var types = VisitTypes("SELECT id FROM customer c pivot ( MAX(Permission) for id in ([1],[2]) ) p");

        types.Should().Contain(SqlType.PivotClause);
    }

    [Test]
    public void Visit_unpivot_should_collect_unpivot_clause()
    {
        var types = VisitTypes("SELECT id FROM customer UNPIVOT (id FOR allcols IN (id,rid,mid) ) up");

        types.Should().Contain(SqlType.UnpivotClause);
    }

    [Test]
    public void Visit_for_xml_path_should_collect_for_xml_path_clause()
    {
        var types = VisitTypes("select id from customer for xml path('')");

        types.Should().Contain(SqlType.ForXmlPathClause);
    }

    [Test]
    public void Visit_for_xml_auto_root_should_collect_auto_clause_and_root_directive()
    {
        var types = VisitTypes("select id from customer for xml auto, ROOT('customer')");

        types.Should().Contain(SqlType.ForXmlAutoClause);
        types.Should().Contain(SqlType.ForXmlRootDirective);
    }

    [Test]
    public void Visit_table_hint_index_should_collect_table_hint_index()
    {
        var types = VisitTypes("select id from customer with(nolock, index(1))");

        types.Should().Contain(SqlType.TableHintIndex);
    }

    [Test]
    public void Visit_foreign_key_constraint_should_collect_table_foreign_key()
    {
        var types = VisitTypes("CREATE TABLE tb1 ( CONSTRAINT [FK1] FOREIGN KEY ([Id]) REFERENCES tb2 ([id2]) )");

        types.Should().Contain(SqlType.TableForeignKey);
    }

    [Test]
    public void Visit_primary_key_constraint_should_collect_constraint()
    {
        var types = VisitTypes("CREATE TABLE tb1 ( [Id] INT NOT NULL CONSTRAINT [pk1] PRIMARY KEY NONCLUSTERED ([Id] ASC) WITH (FILLFACTOR = 90) IDENTITY, [name] NVARCHAR(50) NULL )");

        types.Should().Contain(SqlType.Constraint);
    }

    [Test]
    public void Visit_set_value_statement_should_collect_set_value_statement()
    {
        var types = VisitTypes("set @name = 'test'");

        types.Should().Contain(SqlType.SetValueStatement);
    }

    [Test]
    public void Visit_insert_statement_should_collect_insert_statement()
    {
        var statement = new SqlInsertStatement { TableName = "Users", Columns = ["Id", "Name"] };

        var nodes = new SqlVisitor().Visit(statement);

        nodes.Select(node => node.Expression.SqlType)
            .Should().Contain(SqlType.InsertStatement);
    }

    [Test]
    public void Visit_update_statement_should_collect_update_statement()
    {
        var statement = new SqlUpdateStatement { TableName = "Users" };

        var nodes = new SqlVisitor().Visit(statement);

        nodes.Select(node => node.Expression.SqlType)
            .Should().Contain(SqlType.UpdateStatement);
    }
}
