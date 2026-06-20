using T1.SqlSharp.Expressions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseSetOperationAndCteTest
{
    [Test]
    public void Intersect_select()
    {
        var sql = $"""
                   select id from customer
                   intersect
                   select id from emp
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            Unions =
            [
                new SqlUnionSelect
                {
                    Operator = SqlSetOperator.Intersect,
                    SelectStatement = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "emp" }]
                    }
                }
            ]
        });
    }

    [Test]
    public void Except_select()
    {
        var sql = $"""
                   select id from customer
                   except
                   select id from emp
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            Unions =
            [
                new SqlUnionSelect
                {
                    Operator = SqlSetOperator.Except,
                    SelectStatement = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "emp" }]
                    }
                }
            ]
        });
    }

    [Test]
    public void Union_all_select()
    {
        var sql = $"""
                   select id from customer
                   union all
                   select id from emp
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
            FromSources = [new SqlTableSource { TableName = "customer" }],
            Unions =
            [
                new SqlUnionSelect
                {
                    Operator = SqlSetOperator.UnionAll,
                    SelectStatement = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "emp" }]
                    }
                }
            ]
        });
    }

    [Test]
    public void Cte_single()
    {
        var sql = $"""
                   WITH cte AS (
                     select id from customer
                   )
                   select id from cte
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "cte",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "customer" }]
                    }
                }
            ],
            Statement = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                FromSources = [new SqlTableSource { TableName = "cte" }]
            }
        });
    }

    [Test]
    public void Cte_multiple()
    {
        var sql = $"""
                   WITH c1 AS (
                     select id from a
                   ), c2 AS (
                     select id from b
                   )
                   select id from c2
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "c1",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "a" }]
                    }
                },
                new SqlCommonTableExpression
                {
                    Name = "c2",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "b" }]
                    }
                }
            ],
            Statement = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                FromSources = [new SqlTableSource { TableName = "c2" }]
            }
        });
    }

    [Test]
    public void Cte_with_column_list()
    {
        var sql = $"""
                   WITH cte (a, b) AS (
                     select id, name from customer
                   )
                   select a from cte
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "cte",
                    ColumnNames = ["a", "b"],
                    Query = new SelectStatement
                    {
                        Columns =
                        [
                            new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } },
                            new SelectColumn { Field = new SqlFieldExpr { FieldName = "name" } }
                        ],
                        FromSources = [new SqlTableSource { TableName = "customer" }]
                    }
                }
            ],
            Statement = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "a" } }],
                FromSources = [new SqlTableSource { TableName = "cte" }]
            }
        });
    }
}
