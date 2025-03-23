using FluentAssertions;
using Xunit;

namespace VimSharpLib.Tests;

public class WordsTests
{
    [Fact]
    public void Test()
    {
        var chars = "Hello 閃電 World".ToColoredCharArray();
        var words = chars.QueryWordsIndexList().ToList();
        words.Should().BeEquivalentTo([0, 6, 11]);
    }
}