using SqlSharpLit.Common.ParserLit;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlArithmeticTest
{
    [Test]
    public void METHOD()
    {
        var sql = $"""
                   cast(@score1 as nvarchar(3)) + ':' + cast(@score2 as nvarchar(3)))
                   """;
        var sqlParser = new SqlParser(sql);
        //var rc = sqlParser.ParseArithmeticExpression();
    }
}