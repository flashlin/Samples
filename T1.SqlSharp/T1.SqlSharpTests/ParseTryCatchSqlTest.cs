using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseTryCatchSqlTest
{
    [Test]
    public void Begin_try_catch()
    {
        var sql = "BEGIN TRY SET @x = 1 END TRY BEGIN CATCH SET @x = 2 END CATCH";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTryCatchStatement
        {
            TryStatements =
            [
                new SqlSetValueStatement
                {
                    Name = new SqlFieldExpr { FieldName = "@x" },
                    Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                }
            ],
            CatchStatements =
            [
                new SqlSetValueStatement
                {
                    Name = new SqlFieldExpr { FieldName = "@x" },
                    Value = new SqlValue { SqlType = SqlType.IntValue, Value = "2" }
                }
            ]
        });
    }

    [Test]
    public void Try_catch_wrapping_transaction()
    {
        var sql = "BEGIN TRY BEGIN TRANSACTION COMMIT TRANSACTION END TRY BEGIN CATCH ROLLBACK TRANSACTION END CATCH";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTryCatchStatement
        {
            TryStatements =
            [
                new SqlTransactionStatement { Action = SqlTransactionAction.Begin },
                new SqlTransactionStatement { Action = SqlTransactionAction.Commit }
            ],
            CatchStatements =
            [
                new SqlTransactionStatement { Action = SqlTransactionAction.Rollback }
            ]
        });
    }
}
