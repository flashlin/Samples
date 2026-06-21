using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDbccBulkInsertSqlTest
{
    [Test]
    public void Dbcc_command_only()
    {
        var sql = "DBCC CHECKDB";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDbccStatement
        {
            Command = "CHECKDB"
        });
    }

    [Test]
    public void Dbcc_with_arguments()
    {
        var sql = "DBCC SHRINKFILE (mydb_log, 1)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDbccStatement
        {
            Command = "SHRINKFILE",
            Arguments = ["mydb_log", "1"]
        });
    }

    [Test]
    public void Dbcc_with_arguments_and_options()
    {
        var sql = "DBCC CHECKDB ('mydb') WITH NO_INFOMSGS, ALL_ERRORMSGS";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDbccStatement
        {
            Command = "CHECKDB",
            Arguments = ["'mydb'"],
            Options = ["NO_INFOMSGS", "ALL_ERRORMSGS"]
        });
    }

    [Test]
    public void Bulk_insert_basic()
    {
        var sql = "BULK INSERT Orders FROM 'C:\\data\\orders.csv'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBulkInsertStatement
        {
            TableName = "Orders",
            DataFile = "'C:\\data\\orders.csv'"
        });
    }

    [Test]
    public void Bulk_insert_with_options()
    {
        var sql = "BULK INSERT dbo.Orders FROM 'orders.csv' WITH (FIELDTERMINATOR = ',', ROWTERMINATOR = '\\n')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBulkInsertStatement
        {
            TableName = "dbo.Orders",
            DataFile = "'orders.csv'",
            Options = ["FIELDTERMINATOR = ','", "ROWTERMINATOR = '\\n'"]
        });
    }
}
