using FluentAssertions;
using T1.LinqSqlBuildEx;

namespace T1.LinqBuildExTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void from_select_entity()
    {
        var sql = Sql.From<Customer>()
            .Select(x => x)
            .Build();
        sql.Should().Be("SELECT tb1.Id, tb1.Name, tb1.Birth FROM Customer tb1 WITH(NOLOCK)");
    }
}

public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; }
    public DateTime Birth { get; set; }
}

public class Home
{
    public int Id { get; set; }
    public string Address { get; set; }
}