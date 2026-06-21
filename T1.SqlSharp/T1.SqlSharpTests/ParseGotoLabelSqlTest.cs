using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseGotoLabelSqlTest
{
    [Test]
    public void Goto_statement()
    {
        var sql = "GOTO ErrorHandler";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlGotoStatement { Label = "ErrorHandler" });
    }

    [Test]
    public void Label_statement()
    {
        var sql = "ErrorHandler:";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlLabelStatement { Label = "ErrorHandler" });
    }
}
