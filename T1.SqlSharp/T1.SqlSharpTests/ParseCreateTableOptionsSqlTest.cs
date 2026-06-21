using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateTableOptionsSqlTest
{
    [Test]
    public void Create_table_on_filegroup_and_textimage_on()
    {
        var sql = "CREATE TABLE t (id int) ON UserData TEXTIMAGE_ON ImageData";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.OnFileGroup, Is.EqualTo("UserData"));
        Assert.That(table.TextImageOn, Is.EqualTo("ImageData"));
    }

    [Test]
    public void Create_table_on_partition_scheme()
    {
        var sql = "CREATE TABLE t (id int) ON ps (id)";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.OnFileGroup, Is.EqualTo("ps (id)"));
    }

    [Test]
    public void Create_table_as_select()
    {
        var sql = "CREATE TABLE t2 AS SELECT * FROM t1";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.TableName, Is.EqualTo("t2"));
        Assert.That(table.AsSelect, Is.InstanceOf<SelectStatement>());
    }

    [Test]
    public void Create_table_as_select_with_distribution()
    {
        var sql = "CREATE TABLE t2 WITH (DISTRIBUTION = ROUND_ROBIN) AS SELECT a FROM t1";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.WithOptions, Is.EqualTo(new List<string> { "DISTRIBUTION = ROUND_ROBIN" }));
        Assert.That(table.AsSelect, Is.Not.Null);
    }

    [Test]
    public void Create_table_temporal_period_and_system_versioning()
    {
        var sql = "CREATE TABLE t (id int, PERIOD FOR SYSTEM_TIME (ValidFrom, ValidTo)) WITH (SYSTEM_VERSIONING = ON)";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.Period, Is.EqualTo("ValidFrom, ValidTo"));
        Assert.That(table.WithOptions, Is.EqualTo(new List<string> { "SYSTEM_VERSIONING = ON" }));
    }

    [Test]
    public void Create_table_with_options()
    {
        var sql = "CREATE TABLE t (id int) WITH (MEMORY_OPTIMIZED = ON)";
        var rc = sql.ParseSql();
        var table = rc.ResultValue as SqlCreateTableExpression;
        Assert.That(table, Is.Not.Null);
        Assert.That(table!.WithOptions, Is.EqualTo(new List<string> { "MEMORY_OPTIMIZED = ON" }));
    }
}
