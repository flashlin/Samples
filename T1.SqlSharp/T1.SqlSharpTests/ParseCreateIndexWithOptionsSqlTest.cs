using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateIndexWithOptionsSqlTest
{
    [Test]
    public void Create_index_with_options()
    {
        var sql = "CREATE INDEX IX_Users ON Users (Name) WITH (PAD_INDEX = ON, FILLFACTOR = 80)";
        var rc = sql.ParseSql();
        var result = rc.ResultValue as SqlCreateIndexStatement;
        Assert.That(rc.HasError, Is.False);
        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Options, Is.EqualTo(new[] { "PAD_INDEX = ON", "FILLFACTOR = 80" }));
    }
}
