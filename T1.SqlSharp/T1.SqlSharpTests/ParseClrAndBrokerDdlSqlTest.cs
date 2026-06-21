using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseClrAndBrokerDdlSqlTest
{
    [Test]
    public void Create_assembly_from_file()
    {
        var sql = "CREATE ASSEMBLY asm FROM 'C:\\x.dll' WITH PERMISSION_SET = SAFE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateAssemblyStatement
        {
            Name = "asm",
            From = "'C:\\x.dll'",
            Options = ["PERMISSION_SET = SAFE"]
        });
    }

    [Test]
    public void Create_rule()
    {
        var sql = "CREATE RULE r AS @v > 0";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SqlCreateRuleOrDefaultStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.IsRule, Is.True);
        Assert.That(result.Name, Is.EqualTo("r"));
    }

    [Test]
    public void Create_default()
    {
        var sql = "CREATE DEFAULT d AS 0";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SqlCreateRuleOrDefaultStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.IsRule, Is.False);
        Assert.That(result.Name, Is.EqualTo("d"));
    }

    [Test]
    public void Create_aggregate()
    {
        var sql = "CREATE AGGREGATE agg (@v int) RETURNS int EXTERNAL NAME asm.cls";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateAggregateStatement
        {
            Name = "agg",
            Parameters = ["@v int"],
            ReturnType = "int",
            ExternalName = "asm.cls"
        });
    }

    [Test]
    public void Create_queue()
    {
        var sql = "CREATE QUEUE q WITH (STATUS = ON)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateQueueStatement
        {
            Name = "q",
            Options = ["STATUS = ON"]
        });
    }
}
