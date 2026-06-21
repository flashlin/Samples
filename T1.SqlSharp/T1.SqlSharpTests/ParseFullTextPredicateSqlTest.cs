using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseFullTextPredicateSqlTest
{
    [Test]
    public void Where_contains_predicate()
    {
        var sql = "SELECT Id FROM Docs WHERE CONTAINS(Body, 'sql')";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(rc.HasError, Is.False);
        Assert.That(result!.Where, Is.Not.Null);
        Assert.That(result.Where!.ToSql(), Does.Contain("CONTAINS"));
    }

    [Test]
    public void Where_freetext_predicate()
    {
        var sql = "SELECT Id FROM Docs WHERE FREETEXT(Body, 'sql server')";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(rc.HasError, Is.False);
        Assert.That(result!.Where, Is.Not.Null);
        Assert.That(result.Where!.ToSql(), Does.Contain("FREETEXT"));
    }
}
