using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseControlFlowSqlTest
{
    [Test]
    public void Begin_end_block()
    {
        var sql = "BEGIN DECLARE @x INT SET @x = 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBlockStatement
        {
            Statements =
            [
                new SqlDeclareStatement
                {
                    Declarations = [new SqlVariableDeclaration { Name = "@x", DataType = "INT" }]
                },
                new SqlSetValueStatement
                {
                    Name = new SqlFieldExpr { FieldName = "@x" },
                    Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                }
            ]
        });
    }

    [Test]
    public void If_then()
    {
        var sql = "IF @x > 0 SET @x = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlIfStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Then = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            }
        });
    }

    [Test]
    public void If_then_else()
    {
        var sql = "IF @x > 0 SET @x = 0 ELSE SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlIfStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Then = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Else = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void If_with_begin_block()
    {
        var sql = "IF @x > 0 BEGIN SET @x = 0 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlIfStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Then = new SqlBlockStatement
            {
                Statements =
                [
                    new SqlSetValueStatement
                    {
                        Name = new SqlFieldExpr { FieldName = "@x" },
                        Value = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                    }
                ]
            }
        });
    }

    [Test]
    public void While_loop()
    {
        var sql = "WHILE @x > 0 SET @x = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWhileStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            }
        });
    }
}
