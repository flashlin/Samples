using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterObjectSqlTest
{
    [Test]
    public void Alter_view()
    {
        var sql = "ALTER VIEW vw_active AS SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            IsAlter = true,
            ViewName = "vw_active",
            Query = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Users" }]
            }
        });
    }

    [Test]
    public void Alter_procedure()
    {
        var sql = "ALTER PROCEDURE usp_Run AS BEGIN SET @x = 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            IsAlter = true,
            ProcedureName = "usp_Run",
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
    public void Alter_function_scalar()
    {
        var sql = "ALTER FUNCTION AddOne (@x INT) RETURNS INT AS BEGIN RETURN @x + 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
            IsAlter = true,
            FunctionName = "AddOne",
            Parameters = [new SqlProcedureParameter { Name = "@x", DataType = "INT" }],
            ReturnType = "INT",
            Body = new SqlBlockStatement
            {
                Statements =
                [
                    new SqlReturnStatement
                    {
                        Value = new SqlArithmeticBinaryExpr
                        {
                            Left = new SqlFieldExpr { FieldName = "@x" },
                            Operator = ArithmeticOperator.Add,
                            Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                        }
                    }
                ]
            }
        });
    }

    [Test]
    public void Alter_trigger()
    {
        var sql = "ALTER TRIGGER trg ON Orders AFTER UPDATE AS SET @x = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTriggerStatement
        {
            IsAlter = true,
            TriggerName = "trg",
            TableName = "Orders",
            Timing = SqlTriggerTiming.After,
            Events = [SqlTriggerEvent.Update],
            Body = new SqlSetValueStatement
            {
                Name = new SqlFieldExpr { FieldName = "@x" },
                Value = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }
}
