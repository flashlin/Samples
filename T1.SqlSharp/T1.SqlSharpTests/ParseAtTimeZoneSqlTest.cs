using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAtTimeZoneSqlTest
{
    [Test]
    public void Select_at_time_zone()
    {
        var sql = "SELECT CreatedAt AT TIME ZONE 'UTC' FROM Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlAtTimeZoneExpr
                    {
                        Expression = new SqlFieldExpr { FieldName = "CreatedAt" },
                        TimeZone = new SqlValue { SqlType = SqlType.String, Value = "'UTC'" }
                    }
                }
            ],
            FromSources = [new SqlTableSource { TableName = "Orders" }]
        });
    }
}
