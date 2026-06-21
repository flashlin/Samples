using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateSequenceSqlTest
{
    [Test]
    public void Create_sequence_minimal()
    {
        var sql = "CREATE SEQUENCE OrderSeq";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSequenceStatement { SequenceName = "OrderSeq" });
    }

    [Test]
    public void Create_sequence_with_type_start_increment()
    {
        var sql = "CREATE SEQUENCE dbo.OrderSeq AS INT START WITH 1 INCREMENT BY 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSequenceStatement
        {
            SequenceName = "dbo.OrderSeq",
            DataType = "INT",
            StartWith = new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
            IncrementBy = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
        });
    }

    [Test]
    public void Create_sequence_with_minmax_cycle()
    {
        var sql = "CREATE SEQUENCE s START WITH 10 INCREMENT BY 5 MINVALUE 1 MAXVALUE 100 NO CYCLE CACHE 20";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSequenceStatement
        {
            SequenceName = "s",
            StartWith = new SqlValue { SqlType = SqlType.IntValue, Value = "10" },
            IncrementBy = new SqlValue { SqlType = SqlType.IntValue, Value = "5" },
            MinValue = new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
            MaxValue = new SqlValue { SqlType = SqlType.IntValue, Value = "100" },
            IsNoCycle = true,
            CacheSize = new SqlValue { SqlType = SqlType.IntValue, Value = "20" }
        });
    }

    [Test]
    public void Create_sequence_cycle_no_cache_no_minvalue()
    {
        var sql = "CREATE SEQUENCE s NO MINVALUE CYCLE NO CACHE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSequenceStatement
        {
            SequenceName = "s",
            IsNoMinValue = true,
            IsCycle = true,
            IsNoCache = true
        });
    }
}
