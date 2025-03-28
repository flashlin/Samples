using FluentAssertions;
using Xunit;

namespace VimSharpLib.Tests;

public class WordsTests
{
    [Fact]
    public void ShouldReturnCorrectWordStartPositions()
    {
        var chars = "Hello 閃電 World".ToColoredCharArray();
        var words = chars.QueryWordsIndexList().ToList();
        words.Should().BeEquivalentTo([0, 6, 11]);
    }

    [Fact]
    public void ShouldReturnCorrectWordStartPositionsWithMixedContent()
    {
        var chars = "Hello, 12閃電d *23 Sample".ToColoredCharArray();
        var words = chars.QueryWordsIndexList().ToList();
        words.Should().BeEquivalentTo([0, 5, 7, 15, 16, 19]);
    }
}