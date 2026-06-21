using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDropExtendedSqlTest
{
    [Test]
    public void Drop_certificate()
    {
        "DROP CERTIFICATE c".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "CERTIFICATE", Names = ["c"] });
    }

    [Test]
    public void Drop_assembly()
    {
        "DROP ASSEMBLY a".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "ASSEMBLY", Names = ["a"] });
    }

    [Test]
    public void Drop_credential()
    {
        "DROP CREDENTIAL cred".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "CREDENTIAL", Names = ["cred"] });
    }

    [Test]
    public void Drop_aggregate()
    {
        "DROP AGGREGATE agg".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "AGGREGATE", Names = ["agg"] });
    }

    [Test]
    public void Drop_rule_and_default()
    {
        "DROP RULE r".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "RULE", Names = ["r"] });
        "DROP DEFAULT d".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "DEFAULT", Names = ["d"] });
    }

    [Test]
    public void Drop_broker_objects()
    {
        "DROP QUEUE q".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "QUEUE", Names = ["q"] });
        "DROP SERVICE svc".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "SERVICE", Names = ["svc"] });
        "DROP CONTRACT ctr".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "CONTRACT", Names = ["ctr"] });
        "DROP MESSAGE TYPE mt".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "MESSAGE TYPE", Names = ["mt"] });
    }

    [Test]
    public void Drop_keys()
    {
        "DROP MASTER KEY".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "MASTER KEY" });
        "DROP SYMMETRIC KEY sk".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "SYMMETRIC KEY", Names = ["sk"] });
    }

    [Test]
    public void Drop_partition_objects_and_statistics()
    {
        "DROP PARTITION FUNCTION pf".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "PARTITION FUNCTION", Names = ["pf"] });
        "DROP PARTITION SCHEME ps".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "PARTITION SCHEME", Names = ["ps"] });
        "DROP STATISTICS t.s".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "STATISTICS", Names = ["t.s"] });
    }

    [Test]
    public void Drop_fulltext_and_security_policy()
    {
        "DROP FULLTEXT INDEX ON t".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "FULLTEXT INDEX", OnTable = "t" });
        "DROP FULLTEXT CATALOG ftc".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "FULLTEXT CATALOG", Names = ["ftc"] });
        "DROP SECURITY POLICY sp".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "SECURITY POLICY", Names = ["sp"] });
    }

    [Test]
    public void Drop_external_and_endpoint()
    {
        "DROP EXTERNAL TABLE et".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "EXTERNAL TABLE", Names = ["et"] });
        "DROP EXTERNAL DATA SOURCE ds".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "EXTERNAL DATA SOURCE", Names = ["ds"] });
        "DROP ENDPOINT ep".ParseSql().ShouldBe(new SqlDropStatement { TypeName = "ENDPOINT", Names = ["ep"] });
    }
}
