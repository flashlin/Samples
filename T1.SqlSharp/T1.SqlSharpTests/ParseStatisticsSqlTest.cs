using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseStatisticsSqlTest
{
    [Test]
    public void Create_statistics()
    {
        var sql = "CREATE STATISTICS stat_Name ON dbo.Users (LastName, FirstName)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlStatisticsStatement
        {
            IsCreate = true,
            Name = "stat_Name",
            TableName = "dbo.Users",
            Columns = ["LastName", "FirstName"]
        });
    }

    [Test]
    public void Update_statistics_table_only()
    {
        var sql = "UPDATE STATISTICS dbo.Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlStatisticsStatement
        {
            IsCreate = false,
            TableName = "dbo.Users"
        });
    }

    [Test]
    public void Update_statistics_with_name()
    {
        var sql = "UPDATE STATISTICS dbo.Users stat_Name";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlStatisticsStatement
        {
            IsCreate = false,
            TableName = "dbo.Users",
            Name = "stat_Name"
        });
    }
}
