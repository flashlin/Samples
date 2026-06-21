using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterIndexSqlTest
{
    [Test]
    public void Alter_index_rebuild()
    {
        var sql = "ALTER INDEX IX_Users_Name ON dbo.Users REBUILD";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX_Users_Name",
            TableName = "dbo.Users",
            Action = "REBUILD"
        });
    }

    [Test]
    public void Alter_index_set_options()
    {
        var sql = "ALTER INDEX IX_Users_Name ON dbo.Users SET (ALLOW_PAGE_LOCKS = ON)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX_Users_Name",
            TableName = "dbo.Users",
            Action = "SET",
            Options = ["ALLOW_PAGE_LOCKS = ON"]
        });
    }

    [Test]
    public void Alter_index_all_reorganize()
    {
        var sql = "ALTER INDEX ALL ON Users REORGANIZE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "ALL",
            TableName = "Users",
            Action = "REORGANIZE"
        });
    }

    [Test]
    public void Alter_index_disable()
    {
        var sql = "ALTER INDEX IX_X ON Users DISABLE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX_X",
            TableName = "Users",
            Action = "DISABLE"
        });
    }
}
