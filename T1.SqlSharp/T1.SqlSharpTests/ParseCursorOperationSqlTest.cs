using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCursorOperationSqlTest
{
    [Test]
    public void Open_cursor()
    {
        var sql = "OPEN curUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCursorOperationStatement { Action = SqlCursorOperation.Open, CursorName = "curUsers" });
    }

    [Test]
    public void Close_cursor()
    {
        var sql = "CLOSE curUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCursorOperationStatement { Action = SqlCursorOperation.Close, CursorName = "curUsers" });
    }

    [Test]
    public void Deallocate_cursor()
    {
        var sql = "DEALLOCATE curUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCursorOperationStatement { Action = SqlCursorOperation.Deallocate, CursorName = "curUsers" });
    }

    [Test]
    public void Fetch_next_from_cursor_into_variables()
    {
        var sql = "FETCH NEXT FROM curUsers INTO @id, @name";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlFetchStatement
        {
            Direction = "NEXT",
            CursorName = "curUsers",
            IntoVariables = ["@id", "@name"]
        });
    }

    [Test]
    public void Fetch_from_cursor_without_direction()
    {
        var sql = "FETCH FROM curUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlFetchStatement { CursorName = "curUsers" });
    }

    [Test]
    public void Fetch_absolute_with_row_count()
    {
        var sql = "FETCH ABSOLUTE 5 FROM curUsers INTO @id";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlFetchStatement
        {
            Direction = "ABSOLUTE",
            RowCount = new SqlValue { SqlType = SqlType.IntValue, Value = "5" },
            CursorName = "curUsers",
            IntoVariables = ["@id"]
        });
    }
}
