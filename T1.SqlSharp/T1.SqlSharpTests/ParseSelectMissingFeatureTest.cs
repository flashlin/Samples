using T1.SqlSharp.Expressions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseSelectMissingFeatureTest
{
    [Test]
    public void Having_before_order_by_should_keep_both_clauses()
    {
        var sql = $"""
                   SELECT id
                   FROM customer
                   GROUP BY id
                   HAVING COUNT(1) = 2
                   ORDER BY id
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ],
            GroupBy = new SqlGroupByClause
            {
                Columns = [new SqlFieldExpr { FieldName = "id" }]
            },
            Having = new SqlHavingClause
            {
                Condition = new SqlConditionExpression
                {
                    Left = new SqlFunctionExpression
                    {
                        FunctionName = "COUNT",
                        Parameters = [new SqlValue { SqlType = SqlType.IntValue, Value = "1" }]
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "2" }
                }
            },
            OrderBy = new SqlOrderByClause
            {
                Columns = [new SqlOrderColumn { ColumnName = new SqlFieldExpr { FieldName = "id" } }]
            }
        });
    }

    [Test]
    public void Having_directly_after_table_should_not_be_consumed_as_alias()
    {
        var sql = $"""
                   SELECT id
                   FROM customer
                   HAVING COUNT(1) = 2
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ],
            Having = new SqlHavingClause
            {
                Condition = new SqlConditionExpression
                {
                    Left = new SqlFunctionExpression
                    {
                        FunctionName = "COUNT",
                        Parameters = [new SqlValue { SqlType = SqlType.IntValue, Value = "1" }]
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "2" }
                }
            }
        });
    }

    [Test]
    public void Select_into_table()
    {
        var sql = $"""
                   SELECT id, name
                   INTO newtable
                   FROM customer
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } },
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "name" } }
            ],
            Into = "newtable",
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }

    [Test]
    public void Select_into_temp_table()
    {
        var sql = $"""
                   SELECT id
                   INTO #temp
                   FROM customer
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            Into = "#temp",
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }

    [Test]
    public void Order_by_offset_fetch()
    {
        var sql = $"""
                   SELECT id
                   FROM customer
                   ORDER BY id
                   OFFSET 10 ROWS FETCH NEXT 20 ROWS ONLY
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ],
            OrderBy = new SqlOrderByClause
            {
                Columns = [new SqlOrderColumn { ColumnName = new SqlFieldExpr { FieldName = "id" } }],
                Offset = new SqlValue { SqlType = SqlType.IntValue, Value = "10" },
                Fetch = new SqlValue { SqlType = SqlType.IntValue, Value = "20" }
            }
        });
    }

    [Test]
    public void Order_by_offset_only()
    {
        var sql = $"""
                   SELECT id
                   FROM customer
                   ORDER BY id
                   OFFSET 5 ROWS
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ],
            OrderBy = new SqlOrderByClause
            {
                Columns = [new SqlOrderColumn { ColumnName = new SqlFieldExpr { FieldName = "id" } }],
                Offset = new SqlValue { SqlType = SqlType.IntValue, Value = "5" }
            }
        });
    }

    [Test]
    public void Full_outer_join()
    {
        var sql = $"""
                   select id
                   from customer c
                   full outer join emp e on e.id = c.id
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Full,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "e.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "c.id" }
                    }
                }
            ]
        });
    }

    [Test]
    public void Full_join()
    {
        var sql = $"""
                   select id
                   from customer c
                   full join emp e on e.id = c.id
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Full,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "e.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "c.id" }
                    }
                }
            ]
        });
    }

    [Test]
    public void Cross_join()
    {
        var sql = $"""
                   select id
                   from customer c
                   cross join emp e
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Cross,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = null
                }
            ]
        });
    }

    [Test]
    public void Cross_apply()
    {
        var sql = $"""
                   select id
                   from customer c
                   cross apply fn_items(c.id) t
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.CrossApply,
                    JoinedTable = new SqlFuncTableSource
                    {
                        Function = new SqlFunctionExpression
                        {
                            FunctionName = "fn_items",
                            Parameters = [new SqlFieldExpr { FieldName = "c.id" }]
                        },
                        Alias = "t"
                    },
                    OnCondition = null
                }
            ]
        });
    }

    [Test]
    public void Outer_apply()
    {
        var sql = $"""
                   select id
                   from customer c
                   outer apply fn_items(c.id) t
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.OuterApply,
                    JoinedTable = new SqlFuncTableSource
                    {
                        Function = new SqlFunctionExpression
                        {
                            FunctionName = "fn_items",
                            Parameters = [new SqlFieldExpr { FieldName = "c.id" }]
                        },
                        Alias = "t"
                    },
                    OnCondition = null
                }
            ]
        });
    }
}
