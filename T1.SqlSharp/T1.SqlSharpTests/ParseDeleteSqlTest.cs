using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDeleteSqlTest
{
    [Test]
    public void Delete_from_table()
    {
        var sql = "DELETE FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            TableName = "Users"
        });
    }

    [Test]
    public void Delete_from_table_with_where()
    {
        var sql = "DELETE FROM Users WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            TableName = "Users",
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Delete_without_from_keyword()
    {
        var sql = "DELETE Users WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            TableName = "Users",
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Delete_top_n_with_output()
    {
        var sql = "DELETE TOP (10) FROM Users OUTPUT deleted.Id WHERE Active = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlParenthesizedExpression
                {
                    Inner = new SqlValue { SqlType = SqlType.IntValue, Value = "10" }
                }
            },
            TableName = "Users",
            Output = new SqlOutputClause
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "deleted.Id" } }
                ]
            },
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Active" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            }
        });
    }

    [Test]
    public void Delete_with_table_hint()
    {
        var sql = "DELETE FROM Users WITH (TABLOCK) WHERE Id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            TableName = "Users",
            Withs = [new SqlHint { Name = "TABLOCK" }],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Delete_with_join_from()
    {
        var sql = "DELETE c FROM customer c JOIN emp e ON c.id = e.id WHERE e.active = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeleteStatement
        {
            TableName = "c",
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
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "e.active" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            }
        });
    }
}
