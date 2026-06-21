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
}
