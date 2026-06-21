using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseTransactionSqlTest
{
    [Test]
    public void Begin_transaction()
    {
        var sql = "BEGIN TRANSACTION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Begin });
    }

    [Test]
    public void Begin_tran_with_name()
    {
        var sql = "BEGIN TRAN MyTran";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Begin, Name = "MyTran" });
    }

    [Test]
    public void Commit_transaction()
    {
        var sql = "COMMIT TRANSACTION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Commit });
    }

    [Test]
    public void Commit_bare()
    {
        var sql = "COMMIT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Commit });
    }

    [Test]
    public void Rollback_bare()
    {
        var sql = "ROLLBACK";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Rollback });
    }

    [Test]
    public void Save_transaction_with_name()
    {
        var sql = "SAVE TRANSACTION SavePoint1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Save, Name = "SavePoint1" });
    }

    [Test]
    public void Begin_distributed_transaction()
    {
        var sql = "BEGIN DISTRIBUTED TRANSACTION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Begin, IsDistributed = true });
    }

    [Test]
    public void Begin_transaction_with_mark_description()
    {
        var sql = "BEGIN TRANSACTION WITH MARK 'Daily backup'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement
        {
            Action = SqlTransactionAction.Begin,
            WithMark = true,
            MarkDescription = "'Daily backup'"
        });
    }

    [Test]
    public void Begin_named_transaction_with_mark()
    {
        var sql = "BEGIN TRAN MyTran WITH MARK";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement
        {
            Action = SqlTransactionAction.Begin,
            Name = "MyTran",
            WithMark = true
        });
    }

    [Test]
    public void Begin_tran_with_variable_name()
    {
        var sql = "BEGIN TRAN @t";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement { Action = SqlTransactionAction.Begin, Name = "@t" });
    }

    [Test]
    public void Commit_transaction_with_delayed_durability()
    {
        var sql = "COMMIT TRANSACTION WITH (DELAYED_DURABILITY = ON)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTransactionStatement
        {
            Action = SqlTransactionAction.Commit,
            WithOptions = ["DELAYED_DURABILITY = ON"]
        });
    }
}
