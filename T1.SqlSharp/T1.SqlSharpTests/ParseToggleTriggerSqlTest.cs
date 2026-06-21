using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseToggleTriggerSqlTest
{
    [Test]
    public void Enable_trigger_on_table()
    {
        var sql = "ENABLE TRIGGER trg_Audit ON Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlToggleTriggerStatement
        {
            Enable = true,
            TriggerNames = ["trg_Audit"],
            Target = "Orders"
        });
    }

    [Test]
    public void Disable_all_triggers_on_table()
    {
        var sql = "DISABLE TRIGGER ALL ON Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlToggleTriggerStatement
        {
            Enable = false,
            TriggerNames = ["ALL"],
            Target = "Orders"
        });
    }

    [Test]
    public void Disable_trigger_on_database()
    {
        var sql = "DISABLE TRIGGER trg_DdlAudit ON DATABASE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlToggleTriggerStatement
        {
            Enable = false,
            TriggerNames = ["trg_DdlAudit"],
            Target = "DATABASE"
        });
    }
}
