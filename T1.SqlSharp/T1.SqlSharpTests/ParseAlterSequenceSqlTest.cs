using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterSequenceSqlTest
{
    [Test]
    public void Alter_sequence_restart_with()
    {
        var sql = "ALTER SEQUENCE dbo.OrderSeq RESTART WITH 1000";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterSequenceStatement
        {
            SequenceName = "dbo.OrderSeq",
            IsRestart = true,
            RestartWith = new SqlValue { SqlType = SqlType.IntValue, Value = "1000" }
        });
    }

    [Test]
    public void Alter_sequence_restart_only()
    {
        var sql = "ALTER SEQUENCE OrderSeq RESTART";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterSequenceStatement
        {
            SequenceName = "OrderSeq",
            IsRestart = true
        });
    }

    [Test]
    public void Alter_sequence_increment_by()
    {
        var sql = "ALTER SEQUENCE OrderSeq INCREMENT BY 5";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterSequenceStatement
        {
            SequenceName = "OrderSeq",
            IncrementBy = new SqlValue { SqlType = SqlType.IntValue, Value = "5" }
        });
    }
}
