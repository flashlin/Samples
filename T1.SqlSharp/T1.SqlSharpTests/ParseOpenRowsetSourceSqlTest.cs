using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseOpenRowsetSourceSqlTest
{
    [Test]
    public void Select_from_openjson_with_path()
    {
        var sql = "SELECT value FROM OPENJSON(@json, '$.items')";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.FromSources.Count, Is.EqualTo(1));
    }

    [Test]
    public void Select_from_openquery()
    {
        var sql = "SELECT * FROM OPENQUERY(LinkedSrv, 'SELECT 1')";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.FromSources.Count, Is.EqualTo(1));
    }
}
