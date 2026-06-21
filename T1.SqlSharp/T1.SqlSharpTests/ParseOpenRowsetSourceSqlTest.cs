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
    public void Select_from_openjson_with_schema()
    {
        var sql = "SELECT * FROM OPENJSON(@json) WITH (id int '$.id', name nvarchar(100) '$.name', tags nvarchar(max) '$.tags' AS JSON)";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SelectStatement;
        Assert.That(result, Is.Not.Null);
        var source = result!.FromSources[0] as SqlFuncTableSource;
        Assert.That(source, Is.Not.Null);
        Assert.That(source!.JsonSchemaColumns, Is.EqualTo(new List<string>
        {
            "id int '$.id'",
            "name nvarchar(100) '$.name'",
            "tags nvarchar(MAX) '$.tags' AS JSON"
        }));
    }

    [Test]
    public void Select_from_openrowset_bulk()
    {
        var sql = "SELECT * FROM OPENROWSET(BULK 'f.txt', SINGLE_CLOB) AS x";
        Assert.DoesNotThrow(() => sql.ParseSql());
        var result = sql.ParseSql().ResultValue as SelectStatement;
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
