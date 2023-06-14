using System.Text.Json.Serialization;

namespace LargeUtilsTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        Assert.Pass();
    }
}

public class MyClass
{
    [JsonPropertyName("RefNo")]
    public int Id { get; set; }
    
    public string Name { get; set; }
}