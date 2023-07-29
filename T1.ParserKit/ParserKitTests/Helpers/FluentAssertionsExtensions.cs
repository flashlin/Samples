using FluentAssertions;

namespace ParserKitTests.Helpers;

public static class FluentAssertionsExtensions
{
    public static void AllSatisfy<T>(this ICollection<T> items, ICollection<T> expectedItems)
    {
        foreach (var (item, expected) in items.Zip(expectedItems))
        {
            item.Should().BeOfType(expected!.GetType())
                .And.BeEquivalentTo(expected,
                    options => options.IncludingAllRuntimeProperties());
        }
    }
}