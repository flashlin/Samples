using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseReturnSqlTest
{
    [Test]
    public void Return_no_value()
    {
        var sql = "RETURN";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlReturnStatement());
    }

    [Test]
    public void Return_with_expression()
    {
        var sql = "RETURN @x + 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlReturnStatement
        {
            Value = new SqlArithmeticBinaryExpr
            {
                Left = new SqlFieldExpr { FieldName = "@x" },
                Operator = ArithmeticOperator.Add,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }
}
