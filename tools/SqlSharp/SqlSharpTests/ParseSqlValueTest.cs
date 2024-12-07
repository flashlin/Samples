using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlValueTest
{
    [Test]
    public void As()
    {
        var sql = $"""
                   @score1 as nvarchar(3)
                   """;
        var rc = ParseValue(sql);
        rc.ResultValue.ShouldBe(new SqlFunctionExpression
        {
            FunctionName = "cast",
            Parameters = [
                new SqlAsExpr
                {
                    Value = new SqlFieldExpression
                    {
                        FieldName = "@score1",
                    },
                    DataType = new SqlDataType
                    {
                        DataTypeName = "nvarchar",
                        Size = new SqlDataSize()
                        {
                            Size = "3",
                        },
                    }
                }
            ],
        });
    }

    [Test]
    public void Cast()
    {
        var sql = $"""
                   cast(@score1 as nvarchar(3))
                   """;
        var rc = ParseValue(sql);
        rc.ResultValue.ShouldBe(new SqlFunctionExpression
        {
            FunctionName = "cast",
            Parameters = [
                new SqlAsExpr
                {
                    Value = new SqlFieldExpression
                    {
                        FieldName = "@score1",
                    },
                    DataType = new SqlDataType
                    {
                        DataTypeName = "nvarchar",
                        Size = new SqlDataSize()
                        {
                            Size = "3",
                        },
                    }
                }
            ],
        });
    }

    private static ParseResult<ISqlValue> ParseValue(string sql)
    {
        var sqlParser = new SqlParser(sql);
        var rc = sqlParser.ParseValue();
        return rc;
    }
}