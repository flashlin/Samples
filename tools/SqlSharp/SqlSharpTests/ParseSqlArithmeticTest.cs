using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlBinaryExprTest
{
    [Test]
    public void A_and_b()
    {
        var sql = $"""
                   @a & b
                   """;
        var sqlParser = new SqlParser(sql);
        var rc = sqlParser.ParseArithmeticExpr();
        rc.ResultValue.ShouldBe(new SqlArithmeticBinaryExpr
        {
            Left = new SqlFieldExpression
            {
                FieldName = "@a"
            },
            Operator = ArithmeticOperator.BitwiseAnd,
            Right = new SqlFieldExpression
            {
                FieldName = "b"
            }
        });
    }
}
    


[TestFixture]
public class ParseSqlArithmeticTest
{
    [Test]
    public void Plus()
    {
        var sql = $"""
                   cast(@score1 as nvarchar(3)) + ':' + cast(@score2 as nvarchar(4)))
                   """;
        var sqlParser = new SqlParser(sql);
        var rc = sqlParser.ParseArithmeticExpr();
        rc.ResultValue.ShouldBe(new SqlArithmeticBinaryExpr
        {
            Left = new SqlArithmeticBinaryExpr
            {
                Left = new SqlFunctionExpression
                {
                    FunctionName = "cast",
                    Parameters = [
                        new SqlAsExpr
                        {
                            Instance = new SqlFieldExpression
                            { 
                                FieldName  = "@score1"
                            },
                            DataType = new SqlDataType()
                            {
                                DataTypeName = "nvarchar",
                                Size = new SqlDataSize()
                                {
                                    Size = "3"
                                }
                            }
                        }
                    ],
                },
                Operator = ArithmeticOperator.Add,
                Right = new SqlValue
                {
                    Value = "':'"
                }
            },
            Operator = ArithmeticOperator.Add,
            Right = new SqlFunctionExpression
            {
                FunctionName = "cast",
                Parameters = [
                    new SqlAsExpr
                    {
                        Instance = new SqlFieldExpression
                        { 
                            FieldName  = "@score2"
                        },
                        DataType = new SqlDataType()
                        {
                            DataTypeName = "nvarchar",
                            Size = new SqlDataSize()
                            {
                                Size = "4"
                            }
                        }
                    }
                ],
            }
        });
    }
}