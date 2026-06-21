using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateFunctionSqlTest
{
    [Test]
    public void Create_scalar_function()
    {
        var sql = "CREATE FUNCTION AddOne (@x INT) RETURNS INT AS BEGIN RETURN @x + 1 END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
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
    public void Create_scalar_function_with_size_return()
    {
        var sql = "CREATE FUNCTION F (@s VARCHAR(10)) RETURNS VARCHAR(20) AS BEGIN RETURN @s END";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
            FunctionName = "F",
            Parameters =
            [
                new SqlProcedureParameter { Name = "@s", DataType = "VARCHAR", DataSize = new SqlDataSize { Size = "10" } }
            ],
            ReturnType = "VARCHAR",
            ReturnSize = new SqlDataSize { Size = "20" },
            Body = new SqlBlockStatement
            {
                Statements = [new SqlReturnStatement { Value = new SqlFieldExpr { FieldName = "@s" } }]
            }
        });
    }

    [Test]
    public void Create_or_alter_inline_table_function()
    {
        var sql = "CREATE OR ALTER FUNCTION GetUsers () RETURNS TABLE AS RETURN (SELECT Id FROM Users)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFunctionStatement
        {
            IsOrAlter = true,
            FunctionName = "GetUsers",
            Parameters = [],
            ReturnType = "TABLE",
            Body = new SqlReturnStatement
            {
                Value = new SqlParenthesizedExpression
                {
                    Inner = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                        FromSources = [new SqlTableSource { TableName = "Users" }]
                    }
                }
            }
        });
    }
}
