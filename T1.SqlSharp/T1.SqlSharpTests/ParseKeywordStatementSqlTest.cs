using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseKeywordStatementSqlTest
{
    [Test]
    public void Checkpoint()
    {
        var sql = "CHECKPOINT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "CHECKPOINT" });
    }

    [Test]
    public void Reconfigure()
    {
        var sql = "RECONFIGURE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "RECONFIGURE" });
    }

    [Test]
    public void Revert()
    {
        var sql = "REVERT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "REVERT" });
    }
}
