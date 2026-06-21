using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterObjectGenericSqlTest
{
    [Test]
    public void Alter_assembly()
    {
        "ALTER ASSEMBLY a FROM 'x.dll'".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "ASSEMBLY", Name = "a", Action = "FROM 'x.dll'"
        });
    }

    [Test]
    public void Alter_certificate()
    {
        "ALTER CERTIFICATE c REMOVE PRIVATE KEY".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "CERTIFICATE", Name = "c", Action = "REMOVE PRIVATE KEY"
        });
    }

    [Test]
    public void Alter_queue()
    {
        "ALTER QUEUE q WITH STATUS = OFF".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "QUEUE", Name = "q", Action = "WITH STATUS = OFF"
        });
    }

    [Test]
    public void Alter_endpoint()
    {
        "ALTER ENDPOINT ep STATE = STOPPED".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "ENDPOINT", Name = "ep", Action = "STATE = STOPPED"
        });
    }

    [Test]
    public void Alter_service()
    {
        "ALTER SERVICE svc".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "SERVICE", Name = "svc"
        });
    }

    [Test]
    public void Alter_resource_governor()
    {
        "ALTER RESOURCE GOVERNOR RECONFIGURE".ParseSql().ShouldBe(new SqlAlterObjectStatement
        {
            Kind = "RESOURCE GOVERNOR", Action = "RECONFIGURE"
        });
    }

    [Test]
    public void Alter_partition_function()
    {
        var result = "ALTER PARTITION FUNCTION pf() SPLIT RANGE (100)".ParseSql().ResultValue as SqlAlterObjectStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Kind, Is.EqualTo("PARTITION FUNCTION"));
        Assert.That(result.Name, Is.EqualTo("pf"));
        Assert.That(result.Action, Does.Contain("SPLIT RANGE"));
    }

    [Test]
    public void Alter_partition_scheme()
    {
        var result = "ALTER PARTITION SCHEME ps NEXT USED fg2".ParseSql().ResultValue as SqlAlterObjectStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Kind, Is.EqualTo("PARTITION SCHEME"));
        Assert.That(result.Name, Is.EqualTo("ps"));
        Assert.That(result.Action, Does.Contain("NEXT USED"));
    }

    [Test]
    public void Add_signature()
    {
        "ADD SIGNATURE TO obj BY CERTIFICATE c".ParseSql().ShouldBe(new SqlSignatureStatement
        {
            IsAdd = true, Target = "obj", By = "CERTIFICATE c"
        });
    }

    [Test]
    public void Create_xml_index()
    {
        "CREATE PRIMARY XML INDEX ix ON t (col)".ParseSql().ShouldBe(new SqlCreateIndexStatement
        {
            IsXml = true,
            IsPrimaryXml = true,
            IndexName = "ix",
            TableName = "t",
            Columns = [new SqlConstraintColumn { ColumnName = "col", Order = "" }]
        });
    }
}
