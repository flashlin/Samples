using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseNextValueForSqlTest
{
    [Test]
    public void Select_next_value_for()
    {
        var sql = "SELECT NEXT VALUE FOR dbo.OrderSeq";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlNextValueForExpr { SequenceName = "dbo.OrderSeq" } }
            ]
        });
    }
}
