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
    public void Checkpoint_with_duration()
    {
        var sql = "CHECKPOINT 5";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "CHECKPOINT", Argument = "5" });
    }

    [Test]
    public void Reconfigure_with_override()
    {
        var sql = "RECONFIGURE WITH OVERRIDE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "RECONFIGURE", Argument = "WITH OVERRIDE" });
    }

    [Test]
    public void Revert()
    {
        var sql = "REVERT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKeywordStatement { Keyword = "REVERT" });
    }
}
