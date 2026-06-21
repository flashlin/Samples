using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateTriggerSqlTest
{
    [Test]
    public void Create_after_insert_trigger()
    {
        var sql = "CREATE TRIGGER trg_audit ON Orders AFTER INSERT AS BEGIN SET @x = 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTriggerStatement
        {
            TriggerName = "trg_audit",
            TableName = "Orders",
            Timing = SqlTriggerTiming.After,
            Events = [SqlTriggerEvent.Insert],
            Body = new SqlBlockStatement
            {
                Statements =
                [
                    new SqlSetValueStatement
                    {
                        Name = new SqlFieldExpr { FieldName = "@x" },
                        Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                    }
                ]
            }
        });
    }

    [Test]
    public void Create_for_insert_update_trigger()
    {
        var sql = "CREATE TRIGGER trg ON dbo.Customers FOR INSERT, UPDATE AS SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTriggerStatement
        {
            TriggerName = "trg",
            TableName = "dbo.Customers",
            Timing = SqlTriggerTiming.For,
            Events = [SqlTriggerEvent.Insert, SqlTriggerEvent.Update],
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void Create_or_alter_instead_of_delete_trigger()
    {
        var sql = "CREATE OR ALTER TRIGGER trg ON Orders INSTEAD OF DELETE AS SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTriggerStatement
        {
            IsOrAlter = true,
            TriggerName = "trg",
            TableName = "Orders",
            Timing = SqlTriggerTiming.InsteadOf,
            Events = [SqlTriggerEvent.Delete],
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }
}
