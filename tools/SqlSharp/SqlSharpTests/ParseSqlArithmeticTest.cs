using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace SqlSharpTests;

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
                            Instance = new SqlFieldExpr
                            { 
                                FieldName  = "@score1"
                            },
                            As = new SqlDataTypeWithSize()
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
                        Instance = new SqlFieldExpr
                        { 
                            FieldName  = "@score2"
                        },
                        As = new SqlDataTypeWithSize()
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