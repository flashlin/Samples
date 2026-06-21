using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseLoopControlSqlTest
{
    [Test]
    public void Break_statement()
    {
        var sql = "BREAK";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlLoopControlStatement { Action = SqlLoopControlAction.Break });
    }

    [Test]
    public void Continue_statement()
    {
        var sql = "CONTINUE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlLoopControlStatement { Action = SqlLoopControlAction.Continue });
    }

    [Test]
    public void While_loop_with_break()
    {
        var sql = "WHILE @x > 0 BEGIN BREAK END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWhileStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Body = new SqlBlockStatement
            {
                Statements = [new SqlLoopControlStatement { Action = SqlLoopControlAction.Break }]
            }
        });
    }
}
