using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseUseGoSqlTest
{
    [Test]
    public void Use_database()
    {
        var sql = "USE MyDatabase";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlUseStatement { DatabaseName = "MyDatabase" });
    }

    [Test]
    public void Go_batch_separator()
    {
        var sql = "GO";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlGoStatement());
    }

    [Test]
    public void Go_with_count()
    {
        var sql = "GO 5";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlGoStatement { Count = 5 });
    }
}
