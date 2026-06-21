using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseExecSqlTest
{
    [Test]
    public void Exec_proc_no_args()
    {
        var sql = "EXEC GetUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetUsers"
        });
    }

    [Test]
    public void Execute_proc_no_args()
    {
        var sql = "EXECUTE dbo.GetUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "dbo.GetUsers"
        });
    }

    [Test]
    public void Exec_proc_with_args()
    {
        var sql = "EXEC GetUser 1, 'admin'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetUser",
            Arguments =
            [
                new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                new SqlValue { SqlType = SqlType.String, Value = "'admin'" }
            ]
        });
    }
}
