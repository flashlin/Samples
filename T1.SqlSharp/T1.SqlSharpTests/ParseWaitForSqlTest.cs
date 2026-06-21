using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseWaitForSqlTest
{
    [Test]
    public void Waitfor_delay()
    {
        var sql = "WAITFOR DELAY '00:00:05'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWaitForStatement
        {
            Kind = SqlWaitForKind.Delay,
            Time = new SqlValue { SqlType = SqlType.String, Value = "'00:00:05'" }
        });
    }

    [Test]
    public void Waitfor_time()
    {
        var sql = "WAITFOR TIME '22:00'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWaitForStatement
        {
            Kind = SqlWaitForKind.Time,
            Time = new SqlValue { SqlType = SqlType.String, Value = "'22:00'" }
        });
    }
}
