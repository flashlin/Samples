using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseXmlNamespacesSqlTest
{
    [Test]
    public void With_xmlnamespaces_prefix()
    {
        var sql = "WITH XMLNAMESPACES ('http://x' AS ns1) SELECT 1";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SqlXmlNamespacesStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Namespaces, Is.EqualTo(new List<string> { "'http://x' AS ns1" }));
        Assert.That(result.Statement, Is.InstanceOf<SelectStatement>());
    }

    [Test]
    public void With_xmlnamespaces_default_and_prefix()
    {
        var sql = "WITH XMLNAMESPACES (DEFAULT 'http://d', 'http://x' AS ns1) SELECT 1";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SqlXmlNamespacesStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Namespaces, Is.EqualTo(new List<string>
        {
            "DEFAULT 'http://d'",
            "'http://x' AS ns1"
        }));
    }
}
