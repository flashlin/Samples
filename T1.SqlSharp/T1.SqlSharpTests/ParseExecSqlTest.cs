using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseExecSqlTest
{
    [Test]
    public void Exec_proc_no_args()
    {
        var sql = "EXEC GetUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetUsers"
        });
    }

    [Test]
    public void Execute_proc_no_args()
    {
        var sql = "EXECUTE dbo.GetUsers";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "dbo.GetUsers"
        });
    }

    [Test]
    public void Exec_dynamic_sql_at_linked_server()
    {
        var sql = "EXEC ('SELECT 1') AT LinkedSrv";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            DynamicSql = new SqlValue { SqlType = SqlType.String, Value = "'SELECT 1'" },
            AtLinkedServer = "LinkedSrv"
        });
    }

    [Test]
    public void Exec_with_return_variable()
    {
        var sql = "EXEC @ret = dbo.usp_DoWork 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ReturnVariable = "@ret",
            ProcedureName = "dbo.usp_DoWork",
            Arguments = [new SqlValue { SqlType = SqlType.IntValue, Value = "1" }]
        });
    }

    [Test]
    public void Exec_proc_with_args()
    {
        var sql = "EXEC GetUser 1, 'admin'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetUser",
            Arguments =
            [
                new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                new SqlValue { SqlType = SqlType.String, Value = "'admin'" }
            ]
        });
    }

    [Test]
    public void Exec_dynamic_sql_string()
    {
        var sql = "EXEC ('SELECT * FROM Users')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            DynamicSql = new SqlValue { SqlType = SqlType.String, Value = "'SELECT * FROM Users'" }
        });
    }

    [Test]
    public void Exec_dynamic_sql_variable()
    {
        var sql = "EXECUTE (@sql)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            DynamicSql = new SqlFieldExpr { FieldName = "@sql" }
        });
    }

    [Test]
    public void Exec_proc_with_named_parameters()
    {
        var sql = "EXEC GetUser @id = 1, @name = 'admin'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetUser",
            Arguments =
            [
                new SqlExecArgument
                {
                    ParameterName = "@id",
                    Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                },
                new SqlExecArgument
                {
                    ParameterName = "@name",
                    Value = new SqlValue { SqlType = SqlType.String, Value = "'admin'" }
                }
            ]
        });
    }

    [Test]
    public void Exec_proc_with_named_output_parameter()
    {
        var sql = "EXEC GetCount @total = @result OUTPUT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlExecStatement
        {
            ProcedureName = "GetCount",
            Arguments =
            [
                new SqlExecArgument
                {
                    ParameterName = "@total",
                    Value = new SqlFieldExpr { FieldName = "@result" },
                    IsOutput = true
                }
            ]
        });
    }
}
