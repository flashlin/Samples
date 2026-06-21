using System.Text.RegularExpressions;

namespace T1.SqlSharpE2eParser;

public sealed record TSqlFeatureDefinition(
    string FeatureId,
    string Category,
    string Feature,
    int Priority,
    string[] Patterns,
    string[] CorpusCategories)
{
    private readonly Regex[] _regexes = Patterns
        .Select(pattern => new Regex(pattern, RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled))
        .ToArray();

    public bool IsMatch(string text)
    {
        return _regexes.Any(regex => regex.IsMatch(text));
    }
}
