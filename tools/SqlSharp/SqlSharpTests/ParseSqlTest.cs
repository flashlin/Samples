using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlTest
{
    [Test]
    public void CreateTable()
    {
        var sql = $"""
                   CREATE TABLE Persons (
                   id int,
                   LastName varchar(50),
                   Money decimal(10,3),
                   [name] [int] IDENTITY(1,1) NOT NULL
                   );
                   """;

        var rc = ParseSql(sql);

        ThenSqlStatement(rc, new CreateTableStatement
        {
            TableName = "Persons",
            Columns =
            [
                new ColumnDefinition { ColumnName = "id", DataType = "int" },
                new ColumnDefinition { ColumnName = "LastName", DataType = "varchar", Size = 50 },
                new ColumnDefinition { ColumnName = "Money", DataType = "decimal", Size = 10, Scale = 3 },
                new ColumnDefinition
                {
                    ColumnName = "[name]", DataType = "[int]",
                    Identity = new SqlIdentity()
                    {
                        Seed = 1,
                        Increment = 1,
                    }
                },
            ]
        });
    }

    [Test]
    public void Select()
    {
        var sql = $"""
                   SELECT Id, Name 
                   FROM Persons
                   WHERE Id = 1;
                   """;

        var rc = ParseSql(sql);

        ThenSqlStatement(rc, new SelectStatement
        {
            Columns =
            [
                new SelectColumn() { ColumnName = "Id" },
                new SelectColumn() { ColumnName = "Name" },
            ],
            From = new SelectFrom
            {
                FromTableName = "Persons"
            },
            Where = new SqlWhereExpression
            {
                Left = new SqlFieldExpression
                {
                    FieldName = "Id",
                },
                Operation = "=",
                Right = new SqlIntValueExpression
                {
                    Value = 1
                }
            }
        });
    }

    private static void ThenSqlStatement<T>(Either<ISqlExpression, ParseError> rc, T expectedSqlStatement)
        where T : ISqlExpression
    {
        rc.Switch(statement =>
            {
                var castedStatement = (T)statement;
                castedStatement.Should().BeEquivalentTo(expectedSqlStatement);
            },
            error => throw error
        );
    }

    private static Either<ISqlExpression, ParseError> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        var rc = p.Parse();
        return rc;
    }
}