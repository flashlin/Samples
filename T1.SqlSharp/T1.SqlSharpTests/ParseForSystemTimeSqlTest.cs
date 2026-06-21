using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseForSystemTimeSqlTest
{
    [Test]
    public void For_system_time_as_of()
    {
        var sql = "select id from Employees FOR SYSTEM_TIME AS OF '2021-01-01'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "Employees",
                    ForSystemTime = "AS OF '2021-01-01'"
                }
            ]
        });
    }

    [Test]
    public void For_system_time_all_with_alias()
    {
        var sql = "select id from Employees FOR SYSTEM_TIME ALL AS e";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "Employees",
                    ForSystemTime = "ALL",
                    Alias = "e"
                }
            ]
        });
    }

    [Test]
    public void For_system_time_between_and()
    {
        var sql = "select id from Employees FOR SYSTEM_TIME BETWEEN '2020-01-01' AND '2021-01-01'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "Employees",
                    ForSystemTime = "BETWEEN '2020-01-01' AND '2021-01-01'"
                }
            ]
        });
    }
}
