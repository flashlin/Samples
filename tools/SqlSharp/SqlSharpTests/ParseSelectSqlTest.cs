using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlTest
{
    
    [Test]
    public void CreateTable()
    {
        var sql = $"""
                   CREATE TABLE [dbo].[B2CRebateHistory]
                   (
                       [Id] [INT] NOT NULL IDENTITY(1, 1)
                   )
                   """;
        
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

        rc.ShouldBe(new SelectStatement
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
                    Value = "1"
                }
            }
        });
    }
    
    private static Either<ISqlExpression, ParseError> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        var rc = p.Parse();
        return rc;
    }

}