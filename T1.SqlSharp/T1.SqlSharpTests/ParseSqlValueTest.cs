using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

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
        rc.ResultValue.ShouldBe(new SqlAsExpr
        {
            Instance = new SqlFieldExpr
            {
                FieldName = "@score1",
            },
            As = new SqlDataTypeWithSize
            {
                DataTypeName = "nvarchar",
                Size = new SqlDataSize()
                {
                    Size = "3",
                },
            },   
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
                    Instance = new SqlFieldExpr
                    {
                        FieldName = "@score1",
                    },
                    As = new SqlDataTypeWithSize
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

    private static ParseResult<ISqlExpression> ParseValue(string sql)
    {
        var sqlParser = new SqlParser(sql);
        var rc = sqlParser.Parse_Value_As_DataType();
        return rc;
    }
}