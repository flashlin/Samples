using System.Security.AccessControl;
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
                  LastName varchar(255),
                  FirstName varchar(255),
                  Address varchar(255),
                  City varchar(255)
                  );
                  """;
        var p = new SqlParser(sql);
        var rc = p.ParseCreateTableStatement();
        rc.Match(statement=>
            statement.Should().BeEquivalentTo(new CreateTableStatement()),
            error=>throw error 
            );
    }
}