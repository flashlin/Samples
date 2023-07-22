using T1.ParserKit;

namespace ParserKitTests;

public class InputStreamTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Read()
    {
        var input = new InputStream("abc");
        var token1 = input.Next(2);
        Assert.True(token1.SequenceEqual("ab".AsSpan()));
        
        var token2 = input.Next();
        Assert.True(token2.SequenceEqual("c".AsSpan()));
        
        var token3 = input.Next();
        Assert.True(token3.SequenceEqual(ReadOnlySpan<char>.Empty));
    }
    
    [Test]
    public void Peek()
    {
        var input = new InputStream("abc");
        input.Next(3);
        var token1 = input.Peek();
        Assert.True(token1.SequenceEqual(ReadOnlySpan<char>.Empty));
    }
}