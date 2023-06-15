using System.Text;
using System.Text.Json.Serialization;
using FluentAssertions;
using T1.LargeUtils;
using T1.Standard.Serialization;

namespace LargeUtilsTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public async Task Test1()
    {
        var obj1 = new MyClass
        {
            Id = 123,
            Name = "flash"
        };
        var json = new JsonSerializer().Serialize(obj1);
        
        var stream = new MemoryStream();
        var writer = new StreamWriter(stream, Encoding.UTF8);
        await writer.WriteAsync(json);
        await writer.FlushAsync();
        stream.Position = 0;

        var sut = new LargeStreamProcessor();
        var obj2 = await sut.Read<MyClass>(stream);

        obj2.Should().BeEquivalentTo(obj1);
    }
}

public class MyClass
{
    [JsonPropertyName("RefNo")]
    public int Id { get; set; }
    
    public string Name { get; set; }
}