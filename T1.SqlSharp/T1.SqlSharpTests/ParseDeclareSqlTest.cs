using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDeclareSqlTest
{
    [Test]
    public void Declare_single_variable()
    {
        var sql = "DECLARE @count INT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration { Name = "@count", DataType = "INT" }
            ]
        });
    }

    [Test]
    public void Declare_with_initial_value()
    {
        var sql = "DECLARE @count INT = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "@count",
                    DataType = "INT",
                    InitialValue = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                }
            ]
        });
    }

    [Test]
    public void Declare_multiple_variables_with_size()
    {
        var sql = "DECLARE @a INT, @name VARCHAR(50) = 'x'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration { Name = "@a", DataType = "INT" },
                new SqlVariableDeclaration
                {
                    Name = "@name",
                    DataType = "VARCHAR",
                    DataSize = new SqlDataSize { Size = "50" },
                    InitialValue = new SqlValue { SqlType = SqlType.String, Value = "'x'" }
                }
            ]
        });
    }

    [Test]
    public void Declare_cursor_variable()
    {
        var sql = "DECLARE @c CURSOR";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration { Name = "@c", DataType = "CURSOR", IsCursor = true }
            ]
        });
    }

    [Test]
    public void Declare_cursor_with_options()
    {
        var sql = "DECLARE curUsers CURSOR LOCAL STATIC READ_ONLY FOR SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "curUsers",
                    DataType = "CURSOR",
                    IsCursor = true,
                    CursorOptions = ["LOCAL", "STATIC", "READ_ONLY"],
                    CursorSource = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                        FromSources = [new SqlTableSource { TableName = "Users" }]
                    }
                }
            ]
        });
    }

    [Test]
    public void Declare_cursor_for_select()
    {
        var sql = "DECLARE curUsers CURSOR FOR SELECT Id FROM Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "curUsers",
                    DataType = "CURSOR",
                    IsCursor = true,
                    CursorSource = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                        FromSources = [new SqlTableSource { TableName = "Users" }]
                    }
                }
            ]
        });
    }

    [Test]
    public void Declare_table_variable()
    {
        var sql = "DECLARE @t TABLE (Id INT, Name VARCHAR(50))";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "@t",
                    DataType = "TABLE",
                    IsTable = true,
                    TableColumns =
                    [
                        new SqlColumnDefinition { ColumnName = "Id", DataType = "INT" },
                        new SqlColumnDefinition
                        {
                            ColumnName = "Name",
                            DataType = "VARCHAR",
                            DataSize = new SqlDataSize { Size = "50" }
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void Declare_table_variable_with_inline_constraint()
    {
        var sql = "DECLARE @t TABLE (Id INT, PRIMARY KEY (Id))";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "@t",
                    DataType = "TABLE",
                    IsTable = true,
                    TableColumns =
                    [
                        new SqlColumnDefinition { ColumnName = "Id", DataType = "INT" }
                    ],
                    TableConstraints =
                    [
                        new SqlConstraintPrimaryKeyOrUnique
                        {
                            ConstraintType = "PRIMARY KEY",
                            Clustered = "",
                            Columns =
                            [
                                new SqlConstraintColumn { ColumnName = "Id", Order = "" }
                            ]
                        }
                    ]
                }
            ]
        });
    }
}
