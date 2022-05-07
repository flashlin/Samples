using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class DbccTest : TestBase
{
    public DbccTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }
        
    [Fact]
    public void dbcc_updateusage()
    {
        var sql = @"dbcc updateusage(0,@objname) with no_infomsgs";

        Parse(sql);

        ThenExprShouldBe(@"DBCC UPDATEUSAGE(0, @objname) WITH(NO_INFOMSGS)");
    }
    
    [Fact]
    public void dbcc_inputbuffer()
    {
        var sql = @"dbcc inputbuffer(55) with no_infomsgs";

        Parse(sql);

        ThenExprShouldBe(@"DBCC INPUTBUFFER(55) WITH NO_INFOMSGS");
    }
    
    
    [Fact]
    public void dbcc_loginfo()
    {
        var sql = @"dbcc loginfo(dbname)";

        Parse(sql);

        ThenExprShouldBe(@"DBCC LOGINFO(dbname)");
    }
    
    
    [Fact]
    public void dbcc_sqlperf()
    {
        var sql = @"dbcc sqlperf(dbname)";

        Parse(sql);

        ThenExprShouldBe(@"DBCC SQLPERF(dbname)");
    }
    
    
}