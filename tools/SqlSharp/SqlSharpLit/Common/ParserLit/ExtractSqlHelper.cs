using System.Text;
using System.Text.RegularExpressions;

namespace SqlSharpLit.Common.ParserLit;

public interface IDatabaseNameProvider
{
    string GetDatabaseNameFromPath(string path);
}

public class ExtractSqlHelper
{
    IDatabaseNameProvider _databaseNameProvider;

    public ExtractSqlHelper(IDatabaseNameProvider databaseNameProvider)
    {
        _databaseNameProvider = databaseNameProvider;
    }

    public IEnumerable<string> GetSqlFiles(string folder)
    {
        var files = Directory.GetFiles(folder, "*.sql");
        foreach (var file in files)
        {
            yield return file;
        }

        var subFolders = Directory.GetDirectories(folder);
        foreach (var subFolder in subFolders)
        {
            foreach (var file in GetSqlFiles(subFolder))
            {
                yield return file;
            }
        }
    }

    public IEnumerable<SqlFile> GetSqlContentsFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            var sql = File.ReadAllText(sqlFile);
            Console.WriteLine($"{sqlFile}");
            var sqlExpressions = new SqlParser(sql).Extract().ToList();
            yield return new SqlFile
            {
                FileName = sqlFile,
                Sql = sql,
                DatabaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFile),
                CreateTables = ExtractAllCreateTableFromText(sql).ToList(),
                SqlExpressions = sqlExpressions
            };
        }
    }

    public void WriteCreateTablesFromFolder(string folder, string outputFolder)
    {
        var createTablesFile = Path.Combine(outputFolder, "CreateTables.sql");
        using var fileStream = new FileStream(createTablesFile, FileMode.Create);
        var writer = new StreamWriter(fileStream, Encoding.UTF8);
        var sqlTypes = new[]
        {
            SqlType.CreateTable,
            SqlType.AddExtendedProperty
        };
        foreach (var sqlFile in GetSqlContentsFromFolder(folder))
        {
            if (sqlFile.CreateTables.Count == 0)
            {
                continue;
            }
            
            var createTableSqlExpressions = sqlFile.SqlExpressions
                .Where(x => x.SqlType == SqlType.CreateTable)
                .ToList();

            writer.WriteLine($"-- {sqlFile.FileName}");
            writer.WriteLine($"-- Database: {sqlFile.DatabaseName}");

            if (sqlFile.CreateTables.Count != createTableSqlExpressions.Count)
            {
                writer.WriteLine($"-- Total Create Tables: {sqlFile.CreateTables.Count}");
                writer.WriteLine($"-- SqlExpression: {createTableSqlExpressions.Count}");
                writer.WriteLine("-- No other SQL expressions found");
                var startIndex = Math.Min(sqlFile.CreateTables.Count, createTableSqlExpressions.Count);
                writer.WriteLine("/*");
                for(var i = startIndex; i < sqlFile.CreateTables.Count; i++)
                {
                    writer.WriteLine(sqlFile.CreateTables[i]);
                    writer.WriteLine();
                    writer.WriteLine();
                    writer.WriteLine();
                }
                writer.WriteLine("*/");
            }
            // foreach (var createTable in sqlFile.CreateTables)
            // {
            //     writer.WriteLine(createTable);
            //     writer.WriteLine("\n\n\n");
            // }

            foreach (var sqlExpression in createTableSqlExpressions)
            {
                writer.WriteLine(sqlExpression.ToSql());
            }
            writer.Flush();
        }
    }

    public IEnumerable<string> ExtractAllCreateTableFromText(string text)
    {
        do
        {
            var (createTableSql, remainingText) = ExtractCreateTableFromText(text);
            if (string.IsNullOrEmpty(createTableSql))
            {
                break;
            }

            yield return createTableSql;
            text = remainingText;
        } while (true);
    }

    public (string createTableSql, string remainingText) ExtractCreateTableFromText(string text)
    {
        var (truncatedText, length) = FindCreateTableStart(text);
        if (length == 0)
        {
            return (string.Empty, string.Empty);
        }

        var createTableEnd = FindCreateTableEnd(truncatedText, length);
        if (createTableEnd < 0)
        {
            return (string.Empty, string.Empty);
        }

        var createTableSql = truncatedText.Substring(0, createTableEnd + 1);
        var remainingText = truncatedText.Substring(createTableEnd);
        return (createTableSql, remainingText);
    }

    private int FindCreateTableEnd(string truncatedText, int startOffset)
    {
        var openParenthesisIndex = truncatedText.IndexOf('(', startOffset);
        if (openParenthesisIndex < 0)
        {
            return -1;
        }

        var offset = openParenthesisIndex + 1;
        var openParenthesisCount = 1;
        while (offset < truncatedText.Length)
        {
            var c = truncatedText[offset];
            if (c == '(')
            {
                openParenthesisCount++;
            }
            else if (c == ')')
            {
                openParenthesisCount--;
                if (openParenthesisCount == 0)
                {
                    return offset;
                }
            }

            offset++;
        }

        return -1;
    }


    public (string truncatedText, int length) FindCreateTableStart(string text)
    {
        var pattern = @"\bCREATE\s+TABLE\b";
        var regex = new Regex(pattern, RegexOptions.IgnoreCase);
        var match = regex.Match(text);
        if (match.Success)
        {
            var offset = match.Index;
            var prevLineContent = FindPreviousLineContent(text, offset);
            if (ContainsComment(prevLineContent))
            {
                return (string.Empty, 0);
            }

            return (text.Substring(offset), match.Value.Length);
        }

        return (string.Empty, 0);
    }

    private static string FindPreviousLineContent(string text, int offset)
    {
        if (offset == 0)
        {
            return string.Empty;
        }

        var lastNewLineIndex = text.LastIndexOf('\n', offset - 1);
        if (lastNewLineIndex >= 0)
        {
            return text.Substring(lastNewLineIndex + 1);
        }

        return text;
    }

    private static bool ContainsComment(string lineContent)
    {
        return lineContent.StartsWith("--") || lineContent.Contains("/*");
    }
}

public class SqlFile
{
    public string FileName { get; set; } = string.Empty;
    public string Sql { get; set; } = string.Empty;
    public List<string> CreateTables { get; set; } = [];
    public string DatabaseName { get; set; } = string.Empty;
    public List<ISqlExpression> SqlExpressions { get; set; } = [];
}