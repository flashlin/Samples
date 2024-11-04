namespace SqlSharpTests;

[TestFixture]
public class ParseSqlCreateTableTest
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
}