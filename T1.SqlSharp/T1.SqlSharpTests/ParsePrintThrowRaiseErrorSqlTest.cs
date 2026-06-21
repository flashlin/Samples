using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParsePrintThrowRaiseErrorSqlTest
{
    [Test]
    public void Print_string_literal()
    {
        var sql = "PRINT 'Hello'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPrintStatement
        {
            Value = new SqlValue { SqlType = SqlType.String, Value = "'Hello'" }
        });
    }

    [Test]
    public void Print_variable()
    {
        var sql = "PRINT @msg";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPrintStatement
        {
            Value = new SqlFieldExpr { FieldName = "@msg" }
        });
    }

    [Test]
    public void Throw_bare()
    {
        var sql = "THROW";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlThrowStatement());
    }

    [Test]
    public void Throw_with_arguments()
    {
        var sql = "THROW 50000, 'Custom error', 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlThrowStatement
        {
            ErrorNumber = new SqlValue { SqlType = SqlType.IntValue, Value = "50000" },
            Message = new SqlValue { SqlType = SqlType.String, Value = "'Custom error'" },
            State = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
        });
    }

    [Test]
    public void Raiserror_basic()
    {
        var sql = "RAISERROR ('Error occurred', 16, 1)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlRaiseErrorStatement
        {
            Message = new SqlValue { SqlType = SqlType.String, Value = "'Error occurred'" },
            Severity = new SqlValue { SqlType = SqlType.IntValue, Value = "16" },
            State = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
        });
    }

    [Test]
    public void Raiserror_with_arguments_and_options()
    {
        var sql = "RAISERROR ('Error %d', 16, 1, @num) WITH NOWAIT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlRaiseErrorStatement
        {
            Message = new SqlValue { SqlType = SqlType.String, Value = "'Error %d'" },
            Severity = new SqlValue { SqlType = SqlType.IntValue, Value = "16" },
            State = new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
            Arguments = [new SqlFieldExpr { FieldName = "@num" }],
            Options = ["NOWAIT"]
        });
    }
}
