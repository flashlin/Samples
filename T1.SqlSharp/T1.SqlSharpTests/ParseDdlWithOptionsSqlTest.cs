using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDdlWithOptionsSqlTest
{
    [Test]
    public void Create_view_with_schemabinding()
    {
        var sql = "CREATE VIEW vw WITH SCHEMABINDING AS SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            ViewName = "vw",
            Options = ["SCHEMABINDING"],
            Query = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Users" }]
            }
        });
    }

    [Test]
    public void Create_view_with_multiple_options()
    {
        var sql = "CREATE VIEW vw WITH SCHEMABINDING, ENCRYPTION AS SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            ViewName = "vw",
            Options = ["SCHEMABINDING", "ENCRYPTION"],
            Query = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Users" }]
            }
        });
    }

    [Test]
    public void Create_procedure_with_encryption()
    {
        var sql = "CREATE PROCEDURE usp_Run WITH ENCRYPTION AS BEGIN SET @x = 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "usp_Run",
            Options = ["ENCRYPTION"],
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
    public void Create_function_with_schemabinding()
    {
        var sql = "CREATE FUNCTION AddOne (@x INT) RETURNS INT WITH SCHEMABINDING AS BEGIN RETURN @x END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
            FunctionName = "AddOne",
            Parameters = [new SqlProcedureParameter { Name = "@x", DataType = "INT" }],
            ReturnType = "INT",
            Options = ["SCHEMABINDING"],
            Body = new SqlBlockStatement
            {
                Statements = [new SqlReturnStatement { Value = new SqlFieldExpr { FieldName = "@x" } }]
            }
        });
    }

    [Test]
    public void Create_procedure_with_execute_as_caller()
    {
        var sql = "CREATE PROCEDURE usp_Run WITH EXECUTE AS CALLER AS BEGIN SET @x = 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "usp_Run",
            Options = ["EXECUTE AS CALLER"],
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
    public void Create_function_with_schemabinding_and_returns_null_on_null_input()
    {
        var sql = "CREATE FUNCTION AddOne (@x INT) RETURNS INT WITH SCHEMABINDING, RETURNS NULL ON NULL INPUT AS BEGIN RETURN @x END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
            FunctionName = "AddOne",
            Parameters = [new SqlProcedureParameter { Name = "@x", DataType = "INT" }],
            ReturnType = "INT",
            Options = ["SCHEMABINDING", "RETURNS NULL ON NULL INPUT"],
            Body = new SqlBlockStatement
            {
                Statements = [new SqlReturnStatement { Value = new SqlFieldExpr { FieldName = "@x" } }]
            }
        });
    }

    [Test]
    public void Create_trigger_with_encryption()
    {
        var sql = "CREATE TRIGGER trg ON Orders WITH ENCRYPTION AFTER INSERT AS SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTriggerStatement
        {
            TriggerName = "trg",
            TableName = "Orders",
            Options = ["ENCRYPTION"],
            Timing = SqlTriggerTiming.After,
            Events = [SqlTriggerEvent.Insert],
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }
}
