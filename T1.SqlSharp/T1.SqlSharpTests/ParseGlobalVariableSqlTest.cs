using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseGlobalVariableSqlTest
{
    [Test]
    public void Select_global_variable()
    {
        var sql = "SELECT @@ROWCOUNT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "@@ROWCOUNT" } }]
        });
    }

    [Test]
    public void While_fetch_status_loop()
    {
        var sql = "WHILE @@FETCH_STATUS = 0 SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWhileStatement
        {
            Condition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "@@FETCH_STATUS" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            },
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }
}
