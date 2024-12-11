using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlBinaryExprTest
{
    [Test]
    public void Bitwise_And_Group()
    {
        var sql = $"""
                   BetStatus & (524288 | 16)
                   """;
        var rc = ParseArithemeticExpr(sql);
        rc.ResultValue.ShouldBe(new SqlArithmeticBinaryExpr
        {
            Left = new SqlFieldExpr
            {
                FieldName = "BetStatus"
            },
            Operator = ArithmeticOperator.BitwiseAnd,
            Right = new SqlGroup
            {
                Inner = new SqlArithmeticBinaryExpr
                {
                    Left = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "524288"
                    },
                    Operator = ArithmeticOperator.BitwiseOr,
                    Right = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "16"
                    }
                }
            }
        });
    }
    
    [Test]
    public void A_and_b()
    {
        var sql = $"""
                   @a & b
                   """;
        var rc = ParseArithemeticExpr(sql);
        rc.ResultValue.ShouldBe(new SqlArithmeticBinaryExpr
        {
            Left = new SqlFieldExpr
            {
                FieldName = "@a"
            },
            Operator = ArithmeticOperator.BitwiseAnd,
            Right = new SqlFieldExpr
            {
                FieldName = "b"
            }
        });
    }

    private static ParseResult<ISqlExpression> ParseArithemeticExpr(string sql)
    {
        var sqlParser = new SqlParser(sql);
        var rc = sqlParser.ParseArithmeticExpr();
        return rc;
    }
}