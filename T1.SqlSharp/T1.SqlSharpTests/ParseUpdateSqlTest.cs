using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseUpdateSqlTest
{
    [Test]
    public void Update_set_single_column()
    {
        var sql = "UPDATE Users SET Name = 'John'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Name" },
                    Right = new SqlValue { SqlType = SqlType.String, Value = "'John'" }
                }
            ]
        });
    }

    [Test]
    public void Update_set_multiple_columns()
    {
        var sql = "UPDATE Users SET Name = 'John', Age = 30";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Name" },
                    Right = new SqlValue { SqlType = SqlType.String, Value = "'John'" }
                },
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Age" },
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "30" }
                }
            ]
        });
    }

    [Test]
    public void Update_set_with_where()
    {
        var sql = "UPDATE Users SET Age = 30 WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Age" },
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "30" }
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Update_set_default_keyword()
    {
        var sql = "UPDATE Users SET CreatedAt = DEFAULT WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "CreatedAt" },
                    Right = new SqlDefaultValue()
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Update_top_n()
    {
        var sql = "UPDATE TOP (5) Users SET Age = 30";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlParenthesizedExpression
                {
                    Inner = new SqlValue { SqlType = SqlType.IntValue, Value = "5" }
                }
            },
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Age" },
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "30" }
                }
            ]
        });
    }

    [Test]
    public void Update_with_table_hint()
    {
        var sql = "UPDATE Users WITH (TABLOCK) SET Age = 30";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            Withs = [new SqlHint { Name = "TABLOCK" }],
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Age" },
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "30" }
                }
            ]
        });
    }

    [Test]
    public void Update_with_output()
    {
        var sql = "UPDATE Users SET Age = 30 OUTPUT deleted.Age, inserted.Age WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "Users",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "Age" },
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "30" }
                }
            ],
            Output = new SqlOutputClause
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "deleted.Age" } },
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "inserted.Age" } }
                ]
            },
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Update_set_from_join()
    {
        var sql = "UPDATE c SET c.Name = e.Name FROM customer c JOIN emp e ON c.id = e.id";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUpdateStatement
        {
            TableName = "c",
            SetClauses =
            [
                new SqlAssignExpr
                {
                    Left = new SqlFieldExpr { FieldName = "c.Name" },
                    Right = new SqlFieldExpr { FieldName = "e.Name" }
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "c.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "e.id" }
                    }
                }
            ]
        });
    }
}
