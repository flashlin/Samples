using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseInsertSqlTest
{
    [Test]
    public void Insert_round_trip_values()
    {
        var sql = "INSERT INTO Users (Id, Name) VALUES (1, 'Alice')";
        var rc = sql.ParseSql();
        Assert.That(rc.ResultValue.ToSql(), Is.EqualTo("INSERT INTO Users ([Id], [Name]) VALUES (1, 'Alice')"));
    }

    [Test]
    public void Insert_round_trip_multi_row()
    {
        var sql = "INSERT INTO Users (Id, Name) VALUES (1, 'A'), (2, 'B')";
        var rc = sql.ParseSql();
        Assert.That(rc.ResultValue.ToSql(), Is.EqualTo("INSERT INTO Users ([Id], [Name]) VALUES (1, 'A'), (2, 'B')"));
    }

    [Test]
    public void Insert_round_trip_default_values()
    {
        var sql = "INSERT INTO Users DEFAULT VALUES";
        var rc = sql.ParseSql();
        Assert.That(rc.ResultValue.ToSql(), Is.EqualTo("INSERT INTO Users DEFAULT VALUES"));
    }

    [Test]
    public void Insert_round_trip_select_is_faithful_not_parameterized()
    {
        var sql = "INSERT INTO Logs (Msg) SELECT Text FROM Users";
        var rc = sql.ParseSql();
        var result = rc.ResultValue.ToSql();
        Assert.That(result, Does.StartWith("INSERT INTO Logs ([Msg]) "));
        Assert.That(result, Does.Contain("SELECT"));
        Assert.That(result, Does.Not.Contain("@p"));
    }

    [Test]
    public void Insert_into_with_columns_single_row()
    {
        var sql = "INSERT INTO Users (Id, Name) VALUES (1, 'Alice')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Id", "Name"],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    new SqlValue { SqlType = SqlType.String, Value = "'Alice'" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_multi_row()
    {
        var sql = "INSERT INTO Users (Id, Name) VALUES (1, 'Alice'), (2, 'Bob')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Id", "Name"],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    new SqlValue { SqlType = SqlType.String, Value = "'Alice'" }
                ],
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "2" },
                    new SqlValue { SqlType = SqlType.String, Value = "'Bob'" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_without_column_list()
    {
        var sql = "INSERT INTO Users VALUES (1, 'Alice')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = [],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    new SqlValue { SqlType = SqlType.String, Value = "'Alice'" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_without_into_keyword()
    {
        var sql = "INSERT Users (Id) VALUES (1)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Id"],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_select()
    {
        var sql = "INSERT INTO Logs (Msg) SELECT Name FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Logs",
            Columns = ["Msg"],
            SourceSelect = new SelectStatement
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Name" } }
                ],
                FromSources =
                [
                    new SqlTableSource { TableName = "Users" }
                ]
            }
        });
    }

    [Test]
    public void Insert_into_default_values()
    {
        var sql = "INSERT INTO Users DEFAULT VALUES";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = [],
            IsDefaultValues = true
        });
    }

    [Test]
    public void Insert_top_n_into_select()
    {
        var sql = "INSERT TOP (10) INTO Logs (Msg) SELECT Name FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlParenthesizedExpression
                {
                    Inner = new SqlValue { SqlType = SqlType.IntValue, Value = "10" }
                }
            },
            TableName = "Logs",
            Columns = ["Msg"],
            SourceSelect = new SelectStatement
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Name" } }
                ],
                FromSources =
                [
                    new SqlTableSource { TableName = "Users" }
                ]
            }
        });
    }

    [Test]
    public void Insert_into_with_table_hint()
    {
        var sql = "INSERT INTO Users WITH (TABLOCK) (Id) VALUES (1)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Withs = [new SqlHint { Name = "TABLOCK" }],
            Columns = ["Id"],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_with_output_columns()
    {
        var sql = "INSERT INTO Users (Name) OUTPUT inserted.Id, inserted.Name VALUES ('Alice')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Name"],
            Output = new SqlOutputClause
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "inserted.Id" } },
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "inserted.Name" } }
                ]
            },
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.String, Value = "'Alice'" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_with_output_into()
    {
        var sql = "INSERT INTO Users (Name) OUTPUT inserted.Id AS NewId INTO AuditLog (LogId) VALUES ('Alice')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Name"],
            Output = new SqlOutputClause
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "inserted.Id" }, Alias = "NewId" }
                ],
                IntoTable = "AuditLog",
                IntoColumns = ["LogId"]
            },
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.String, Value = "'Alice'" }
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_values_with_default_keyword()
    {
        var sql = "INSERT INTO Users (Id, CreatedAt) VALUES (1, DEFAULT)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Users",
            Columns = ["Id", "CreatedAt"],
            ValuesRows =
            [
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    new SqlDefaultValue()
                ]
            ]
        });
    }

    [Test]
    public void Insert_into_exec_no_args()
    {
        var sql = "INSERT INTO Logs (Msg) EXEC GetMessages";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Logs",
            Columns = ["Msg"],
            ExecSource = new SqlExecStatement { ProcedureName = "GetMessages" }
        });
    }

    [Test]
    public void Insert_into_exec_with_args()
    {
        var sql = "INSERT INTO Logs EXEC GetMessages 1, 'x'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Logs",
            Columns = [],
            ExecSource = new SqlExecStatement
            {
                ProcedureName = "GetMessages",
                Arguments =
                [
                    new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    new SqlValue { SqlType = SqlType.String, Value = "'x'" }
                ]
            }
        });
    }

    [Test]
    public void Insert_into_with_expression_values()
    {
        var sql = "INSERT INTO Logs (CreatedAt, Note) VALUES (GETDATE(), NULL)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlInsertStatement
        {
            TableName = "Logs",
            Columns = ["CreatedAt", "Note"],
            ValuesRows =
            [
                [
                    new SqlFunctionExpression { FunctionName = "GETDATE", Parameters = [] },
                    new SqlNullValue()
                ]
            ]
        });
    }
}
