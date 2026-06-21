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

    [Test]
    public void Create_database_with_size_units()
    {
        var sql = "CREATE DATABASE Sales ON PRIMARY " +
                  "(NAME = s, FILENAME = 'c:\\s.mdf', SIZE = 10MB, MAXSIZE = 50MB, FILEGROWTH = 5MB)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            OnPrimary = true,
            DataFiles = ["(NAME = s, FILENAME = 'c:\\s.mdf', SIZE = 10MB, MAXSIZE = 50MB, FILEGROWTH = 5MB)"]
        });
    }

    [Test]
    public void Create_database_with_filegroup()
    {
        var sql = "CREATE DATABASE Sales ON PRIMARY (NAME = s, FILENAME = 'c:\\s.mdf'), " +
                  "FILEGROUP fg1 (NAME = f1, FILENAME = 'c:\\f1.ndf')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            OnPrimary = true,
            DataFiles = ["(NAME = s, FILENAME = 'c:\\s.mdf')"],
            FileGroups =
            [
                new SqlDatabaseFileGroup
                {
                    Name = "fg1",
                    Files = ["(NAME = f1, FILENAME = 'c:\\f1.ndf')"]
                }
            ]
        });
    }

    [Test]
    public void Create_database_with_containment()
    {
        var sql = "CREATE DATABASE Sales CONTAINMENT = PARTIAL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            Containment = "PARTIAL"
        });
    }

    [Test]
    public void Create_database_with_options()
    {
        var sql = "CREATE DATABASE Sales WITH TRUSTWORTHY ON, DB_CHAINING OFF";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement
        {
            DatabaseName = "Sales",
            Options = ["TRUSTWORTHY ON", "DB_CHAINING OFF"]
        });
    }
}
