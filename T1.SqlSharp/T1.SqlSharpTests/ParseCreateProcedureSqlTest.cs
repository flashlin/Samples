using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateProcedureSqlTest
{
    [Test]
    public void Create_procedure_no_params()
    {
        var sql = "CREATE PROCEDURE GetUsers AS SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "GetUsers",
            Body = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Users" }]
            }
        });
    }

    [Test]
    public void Create_proc_with_param()
    {
        var sql = "CREATE PROC GetUser @id INT AS SELECT Name FROM Users WHERE Id = @id";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "GetUser",
            Parameters =
            [
                new SqlProcedureParameter { Name = "@id", DataType = "INT" }
            ],
            Body = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Name" } }],
                FromSources = [new SqlTableSource { TableName = "Users" }],
                Where = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "Id" },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlFieldExpr { FieldName = "@id" }
                }
            }
        });
    }

    [Test]
    public void Create_proc_params_in_parens()
    {
        var sql = "CREATE PROCEDURE p (@a INT) AS SELECT 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "p",
            Parameters =
            [
                new SqlProcedureParameter { Name = "@a", DataType = "INT" }
            ],
            Body = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlValue { SqlType = SqlType.IntValue, Value = "1" } }]
            }
        });
    }

    [Test]
    public void Create_proc_multiple_bare_statements_body()
    {
        var sql = "CREATE PROCEDURE p AS SET NOCOUNT ON SELECT 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            ProcedureName = "p",
            Body = new SqlBlockStatement
            {
                IsImplicit = true,
                Statements =
                [
                    new SqlSetOptionStatement { Option = "NOCOUNT", Value = "ON" },
                    new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlValue { SqlType = SqlType.IntValue, Value = "1" } }]
                    }
                ]
            }
        });
    }

    [Test]
    public void Create_or_alter_proc_with_default_output_and_block_body()
    {
        var sql = "CREATE OR ALTER PROCEDURE Upd @id INT, @name VARCHAR(50) = 'x' OUTPUT "
                  + "AS BEGIN UPDATE Users SET Name = @name WHERE Id = @id END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateProcedureStatement
        {
            IsOrAlter = true,
            ProcedureName = "Upd",
            Parameters =
            [
                new SqlProcedureParameter { Name = "@id", DataType = "INT" },
                new SqlProcedureParameter
                {
                    Name = "@name",
                    DataType = "VARCHAR",
                    DataSize = new SqlDataSize { Size = "50" },
                    DefaultValue = new SqlValue { SqlType = SqlType.String, Value = "'x'" },
                    IsOutput = true
                }
            ],
            Body = new SqlBlockStatement
            {
                Statements =
                [
                    new SqlUpdateStatement
                    {
                        TableName = "Users",
                        SetClauses =
                        [
                            new SqlAssignExpr
                            {
                                Left = new SqlFieldExpr { FieldName = "Name" },
                                Right = new SqlFieldExpr { FieldName = "@name" }
                            }
                        ],
                        Where = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr { FieldName = "Id" },
                            ComparisonOperator = ComparisonOperator.Equal,
                            Right = new SqlFieldExpr { FieldName = "@id" }
                        }
                    }
                ]
            }
        });
    }
}
