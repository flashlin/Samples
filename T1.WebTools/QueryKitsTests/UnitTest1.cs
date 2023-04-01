using QueryKits.Services;

namespace QueryKitsTests;

public class Tests
{
    private ReportDbContext _sut = null!;

    [SetUp]
    public void Setup()
    {
        _sut = new ReportDbContext(new SqliteMemoryDbContextOptionsFactory());
    }

    [Test]
    public void Test1()
    {
        
        Assert.Pass();
    }
}