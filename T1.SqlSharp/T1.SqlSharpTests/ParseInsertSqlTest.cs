using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseInsertSqlTest
{
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
