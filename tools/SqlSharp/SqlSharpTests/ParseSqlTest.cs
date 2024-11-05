using FluentAssertions;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharpTests;

[TestFixture]
public class ParseSqlTest
{

    [Test]
    public void Test()
    {
        var sql = $"""
                  CREATE TABLE Persons (
                  PersonID int,
                  LastName varchar(50),
                  FirstName varchar(25),
                  Address varchar(255),
                  Money decimal(10,3)
                  );
                  """;
        var p = new SqlParser(sql);
        var rc = p.ParseCreateTableStatement();
        rc.Match(statement=>
            statement.Should().BeEquivalentTo(new CreateTableStatement
            {
                TableName = "Persons",
                Columns =
                [
                    new ColumnDefinition { ColumnName = "PersonID", DataType = "int" },
                    new ColumnDefinition { ColumnName = "LastName", DataType = "varchar", Size = 50 },
                    new ColumnDefinition { ColumnName = "FirstName", DataType = "varchar", Size = 25 },
                    new ColumnDefinition { ColumnName = "Address", DataType = "varchar", Size = 255 },
                    new ColumnDefinition { ColumnName = "Money", DataType = "decimal", Size = 10, Scale = 3 }
                ]
            }),
            error=>throw error 
            );
    }
}