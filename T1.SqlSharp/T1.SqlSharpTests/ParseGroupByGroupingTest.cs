using T1.SqlSharp.Expressions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseGroupByGroupingTest
{
    [Test]
    public void Group_by_rollup()
    {
        var sql = $"""
                   select id, name from customer
                   group by rollup(id, name)
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } },
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "name" } }
            ],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            GroupBy = new SqlGroupByClause
            {
                GroupingType = GroupingType.Rollup,
                Columns =
                [
                    new SqlFieldExpr { FieldName = "id" },
                    new SqlFieldExpr { FieldName = "name" }
                ]
            }
        });
    }

    [Test]
    public void Group_by_cube()
    {
        var sql = $"""
                   select id, name from customer
                   group by cube(id, name)
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } },
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "name" } }
            ],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            GroupBy = new SqlGroupByClause
            {
                GroupingType = GroupingType.Cube,
                Columns =
                [
                    new SqlFieldExpr { FieldName = "id" },
                    new SqlFieldExpr { FieldName = "name" }
                ]
            }
        });
    }

    [Test]
    public void Group_by_grouping_sets()
    {
        var sql = $"""
                   select id, name from customer
                   group by grouping sets ((id, name), (id), ())
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } },
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "name" } }
            ],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            GroupBy = new SqlGroupByClause
            {
                GroupingType = GroupingType.GroupingSets,
                GroupingSets =
                [
                    new SqlGroupingSet
                    {
                        Columns =
                        [
                            new SqlFieldExpr { FieldName = "id" },
                            new SqlFieldExpr { FieldName = "name" }
                        ]
                    },
                    new SqlGroupingSet
                    {
                        Columns = [new SqlFieldExpr { FieldName = "id" }]
                    },
                    new SqlGroupingSet
                    {
                        Columns = []
                    }
                ]
            }
        });
    }
}
