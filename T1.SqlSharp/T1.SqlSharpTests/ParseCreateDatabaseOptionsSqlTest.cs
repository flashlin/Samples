using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateDatabaseOptionsSqlTest
{
    [Test]
    public void Create_database_with_collate()
    {
        var sql = "CREATE DATABASE Sales COLLATE Latin1_General_CI_AS";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            Collation = "Latin1_General_CI_AS"
        });
    }

    [Test]
    public void Create_database_with_on_primary_and_log_on()
    {
        var sql = "CREATE DATABASE Sales ON PRIMARY (NAME = Sales_dat, FILENAME = 'C:\\sales.mdf') " +
                  "LOG ON (NAME = Sales_log, FILENAME = 'C:\\sales.ldf')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            OnPrimary = true,
            DataFiles = ["(NAME = Sales_dat, FILENAME = 'C:\\sales.mdf')"],
            LogFiles = ["(NAME = Sales_log, FILENAME = 'C:\\sales.ldf')"]
        });
    }
}
