using System.Text;
using System.Text.RegularExpressions;

namespace T1.SqlSharpE2eParser;

public sealed class TestSqlSnippetExtractor
{
    private static readonly Regex SqlKeywordRegex = new(
        @"\b(SELECT|CREATE|ALTER|DROP|DECLARE|INSERT|UPDATE|DELETE|MERGE|EXEC|EXECUTE|SET|IF|WHILE|BEGIN|GRANT|DENY|REVOKE|TRUNCATE|WAITFOR|RETURN|WITH)\b",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public IReadOnlyList<TestSqlSnippet> Extract(string testPath)
    {
        return Directory
            .EnumerateFiles(testPath, "*.cs", SearchOption.AllDirectories)
            .SelectMany(ExtractFromFile)
            .Where(IsSqlSnippet)
            .ToList();
    }

    private static IEnumerable<TestSqlSnippet> ExtractFromFile(string filePath)
    {
        var text = File.ReadAllText(filePath);
        foreach (var literal in ExtractRawStringLiterals(text))
        {
            yield return new TestSqlSnippet
            {
                FilePath = filePath,
                LineNumber = GetLineNumber(text, literal.Start),
                Sql = NormalizeSql(literal.Content)
            };
        }
    }

    private static IEnumerable<RawStringLiteral> ExtractRawStringLiterals(string text)
    {
        var index = 0;
        while (index < text.Length)
        {
            if (text[index] != '"')
            {
                index++;
                continue;
            }

            var quoteCount = CountQuotes(text, index);
            if (quoteCount < 3)
            {
                index += quoteCount;
                continue;
            }

            var contentStart = index + quoteCount;
            var closeIndex = FindClosingQuotes(text, contentStart, quoteCount);
            if (closeIndex < 0)
            {
                yield break;
            }

            yield return new RawStringLiteral
            {
                Start = index,
                Content = text[contentStart..closeIndex]
            };

            index = closeIndex + quoteCount;
        }
    }

    private static int CountQuotes(string text, int index)
    {
        var count = 0;
        while (index + count < text.Length && text[index + count] == '"')
        {
            count++;
        }

        return count;
    }

    private static int FindClosingQuotes(string text, int start, int quoteCount)
    {
        var index = start;
        while (index < text.Length)
        {
            if (text[index] != '"')
            {
                index++;
                continue;
            }

            var currentQuoteCount = CountQuotes(text, index);
            if (currentQuoteCount >= quoteCount)
            {
                return index;
            }

            index += currentQuoteCount;
        }

        return -1;
    }

    private static bool IsSqlSnippet(TestSqlSnippet snippet)
    {
        return snippet.Sql.Length >= 12 && SqlKeywordRegex.IsMatch(snippet.Sql);
    }

    private static string NormalizeSql(string sql)
    {
        var lines = sql.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        var nonEmptyLines = lines.Where(line => !string.IsNullOrWhiteSpace(line)).ToList();
        if (nonEmptyLines.Count == 0)
        {
            return string.Empty;
        }

        var minIndent = nonEmptyLines.Min(CountIndent);
        var builder = new StringBuilder();
        foreach (var line in lines)
        {
            var normalizedLine = line.Length >= minIndent ? line[minIndent..] : line.TrimStart();
            builder.AppendLine(normalizedLine.TrimEnd());
        }

        return builder.ToString().Trim();
    }

    private static int CountIndent(string line)
    {
        var count = 0;
        while (count < line.Length && char.IsWhiteSpace(line[count]))
        {
            count++;
        }

        return count;
    }

    private static int GetLineNumber(string text, int offset)
    {
        var line = 1;
        for (var i = 0; i < offset && i < text.Length; i++)
        {
            if (text[i] == '\n')
            {
                line++;
            }
        }

        return line;
    }
}

public sealed record TestSqlSnippet
{
    public required string FilePath { get; init; }
    public required int LineNumber { get; init; }
    public required string Sql { get; init; }
}

public sealed record RawStringLiteral
{
    public required int Start { get; init; }
    public required string Content { get; init; }
}
