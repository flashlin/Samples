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

    [Test]
    public void Alter_index_rebuild_with_options()
    {
        var sql = "ALTER INDEX IX ON dbo.t REBUILD WITH (ONLINE = ON, FILLFACTOR = 80)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX",
            TableName = "dbo.t",
            Action = "REBUILD",
            Options = ["ONLINE = ON", "FILLFACTOR = 80"]
        });
    }

    [Test]
    public void Alter_index_rebuild_partition()
    {
        var sql = "ALTER INDEX IX ON dbo.t REBUILD PARTITION = 3";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX",
            TableName = "dbo.t",
            Action = "REBUILD",
            Partition = "3"
        });
    }

    [Test]
    public void Alter_index_reorganize_with_options()
    {
        var sql = "ALTER INDEX IX ON dbo.t REORGANIZE WITH (LOB_COMPACTION = ON)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterIndexStatement
        {
            IndexName = "IX",
            TableName = "dbo.t",
            Action = "REORGANIZE",
            Options = ["LOB_COMPACTION = ON"]
        });
    }
}
