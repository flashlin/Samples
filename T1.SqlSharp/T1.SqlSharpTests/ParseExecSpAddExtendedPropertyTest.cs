using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseExecSpAddExtendedPropertyTest
{
    [Test]
    public void SysSpAddExtendedProperty()
    {
        var sql = $"""
                   EXEC sys.sp_addextendedproperty @name=N'CreatedBy', @value=N'created by', @level0type=N'SCHEMA', @level0name=N'dbo', @level1type=N'TABLE', @level1name=N'Customer'
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlSpAddExtendedPropertyExpression
        {
            Name = "N'CreatedBy'",
            Value = "N'created by'",
            Level0Type = "N'SCHEMA'",
            Level0Name = "N'dbo'",
            Level1Type = "N'TABLE'",
            Level1Name = "N'Customer'"
        });
    }
    
    [Test]
    public void AddExtendedProperty()
    {
        var sql = $"""
                   EXEC sp_addextendedproperty
                   @name = N'MS_Description',        -- 屬性名稱（固定為 MS_Description 用於說明）
                   @value = N'hello',                -- 說明內容 
                   @level0type = N'SCHEMA',          -- 第 1 級目標類型
                   @level0name = N'dbo',             -- 第 1 級名稱（預設 schema）
                   @level1type = N'TABLE',           -- 第 2 級目標類型
                   @level1name = N'customer',        -- 第 2 級名稱（資料表名稱）
                   @level2type = N'COLUMN',          -- 第 3 級目標類型
                   @level2name = 'addr';            -- 第 3 級名稱（欄位名稱)
                   """;

        var rc = ParseSql(sql);

        rc.ShouldBe(new SqlSpAddExtendedPropertyExpression()
        {
            Name = "N'MS_Description'",
            Value = "N'hello'",
            Level0Type = "N'SCHEMA'",
            Level0Name = "N'dbo'",
            Level1Type = "N'TABLE'",
            Level1Name = "N'customer'",
            Level2Type = "N'COLUMN'",
            Level2Name = "'addr'",
        });
    }
    
    [Test]
    public void Desc()
    {
        var sql = $"""
                   EXEC sp_addextendedproperty N'MS_Description', N'Monday = 1, Tuesday = 2', 'SCHEMA', N'dbo', 'TABLE', N'MySetting', 'COLUMN', N'Setting1'
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SqlSpAddExtendedPropertyExpression
        {
            Name = "N'MS_Description'",
            Value = "N'Monday = 1, Tuesday = 2'",
            Level0Type = "'SCHEMA'",
            Level0Name = "N'dbo'",
            Level1Type = "'TABLE'",
            Level1Name = "N'MySetting'",
            Level2Type = "'COLUMN'",
            Level2Name = "N'Setting1'"
        });
    }
    
    private static ParseResult<ISqlExpression> ParseSql(string sql)
    {
        return SqlParser.Parse(sql);
    }

}