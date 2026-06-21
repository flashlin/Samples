using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseSetOptionSqlTest
{
    [Test]
    public void Set_nocount_on()
    {
        var sql = "SET NOCOUNT ON";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetOptionStatement { Option = "NOCOUNT", Value = "ON" });
    }

    [Test]
    public void Set_xact_abort_off()
    {
        var sql = "SET XACT_ABORT OFF";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetOptionStatement { Option = "XACT_ABORT", Value = "OFF" });
    }

    [Test]
    public void Set_identity_insert_with_table()
    {
        var sql = "SET IDENTITY_INSERT dbo.Users ON";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetOptionStatement { Option = "IDENTITY_INSERT", Target = "dbo.Users", Value = "ON" });
    }

    [Test]
    public void Set_rowcount_number()
    {
        var sql = "SET ROWCOUNT 100";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetOptionStatement { Option = "ROWCOUNT", Value = "100" });
    }

    [Test]
    public void Set_dateformat_identifier()
    {
        var sql = "SET DATEFORMAT mdy";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetOptionStatement { Option = "DATEFORMAT", Value = "mdy" });
    }

    [Test]
    public void Set_variable_assignment_still_works()
    {
        var sql = "SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetValueStatement
        {
            Name = new SqlFieldExpr { FieldName = "@x" },
            Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
        });
    }
}
