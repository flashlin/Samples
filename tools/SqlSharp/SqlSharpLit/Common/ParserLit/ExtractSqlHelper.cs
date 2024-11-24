using System.Text;
using System.Text.RegularExpressions;
using T1.Standard.Collections.Generics;

namespace SqlSharpLit.Common.ParserLit;

public interface IDatabaseNameProvider
{
    string GetDatabaseNameFromPath(string path);
}

public class ExtractSqlHelper
{
    private readonly IDatabaseNameProvider _databaseNameProvider;

    public ExtractSqlHelper(IDatabaseNameProvider databaseNameProvider)
    {
        _databaseNameProvider = databaseNameProvider;
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

    public void GenerateRagFiles(string sqlFolder)
    {
        var sqlFileContents = GetSqlContentsFromFolder(sqlFolder);
        GenerateRagFilesFromSqlContents(sqlFileContents);
    }

    public void GenerateRagFilesFromSqlContents(IEnumerable<SqlFileContent> sqlFileContents)
    {
        var databaseDescriptions = ExtractDatabaseDescriptions(sqlFileContents)
            .ToList();
        var databaseDescriptionMdFile = Path.Combine("outputs", "DatabaseDesc.md");
        using var writer = new StreamWriter(databaseDescriptionMdFile, false, Encoding.UTF8);
        writer.WriteLine(
            "The following is a detailed description of all databases and table structures of Titan Company.");
        foreach (var database in databaseDescriptions)
        {
            foreach (var table in database.Tables)
            {
                writer.WriteLine($"Database Name: {database.DatabaseName}");
                writer.WriteLine(
                    $"The following is a detailed description of all column structures in the {table.TableName} table.");
                foreach (var column in table.Columns)
                {
                    writer.Write($"{column.ColumnName} {column.DataType}");
                    if (column.IsNullable)
                    {
                        writer.Write($" ,is Nullable");
                    }
                    if (column.IsIdentity)
                    {
                        writer.Write($" ,is Identity");
                    }
                    if (column.DefaultValue != string.Empty)
                    {
                        writer.Write($" ,Default Value: {column.DefaultValue}");
                    }

                    if (!string.IsNullOrEmpty(column.Description.Trim()))
                    {
                        writer.Write($" ,Description: {column.Description}");
                    }

                    writer.WriteLine();
                }

                writer.WriteLine();
                writer.WriteLine();
            }
        }

        writer.Flush();
    }

    public IEnumerable<SqlCreateTablesSqlFiles> GetCreateCreateTableSqlFromFolder(
        IEnumerable<SqlFileContent> sqlContents)
    {
        foreach (var sqlFileContent in sqlContents)
        {
            Console.WriteLine($"{sqlFileContent.FileName} = {sqlFileContent.SqlExpressions.Count}");
            var createTablesSql = ExtractAllCreateTableFromText(sqlFileContent.Sql).ToList();
            yield return new SqlCreateTablesSqlFiles
            {
                File = sqlFileContent,
                DatabaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName),
                CreateTables = createTablesSql,
            };
        }
    }

    public IEnumerable<SqlFileContent> GetSqlContentsFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            var sql = File.ReadAllText(sqlFile);
            var sqlExpressions = new SqlParser(sql).Extract().ToList();
            yield return new SqlFileContent
            {
                FileName = sqlFile,
                Sql = sql,
                SqlExpressions = sqlExpressions
            };
        }
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

    public void WriteCreateTablesFromFolder(string folder, string outputFolder)
    {
        if (!Directory.Exists(folder))
        {
            return;
        }

        using var writer = CreateStreamWriter(Path.Combine(outputFolder, "CreateTables.sql"));
        var sqlFileContents = GetSqlContentsFromFolder(folder)
            .ToList();
        WriteCreateTablesTo(sqlFileContents, writer);
        GenerateRagFilesFromSqlContents(sqlFileContents);
    }

    private void WriteCreateTablesTo(IEnumerable<SqlFileContent> sqlFileContents, StreamWriter writer)
    {
        var sqlCreateTables = GetCreateCreateTableSqlFromFolder(sqlFileContents);
        foreach (var sqlFile in sqlCreateTables)
        {
            if (sqlFile.CreateTables.Count == 0)
            {
                continue;
            }

            var createTableSqlExpressions = sqlFile.File.SqlExpressions
                .Where(x => x.SqlType == SqlType.CreateTable)
                .ToList();

            writer.WriteLine($"-- {sqlFile.File.FileName}");
            writer.WriteLine($"-- Database: {sqlFile.DatabaseName}");

            if (sqlFile.CreateTables.Count != createTableSqlExpressions.Count)
            {
                writer.WriteLine($"-- Total Create Tables: {sqlFile.CreateTables.Count}");
                writer.WriteLine($"-- SqlExpression: {createTableSqlExpressions.Count}");
                writer.WriteLine("-- No other SQL expressions found");
                var startIndex = Math.Min(sqlFile.CreateTables.Count, createTableSqlExpressions.Count);
                writer.WriteLine("/*");
                for (var i = startIndex; i < sqlFile.CreateTables.Count; i++)
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

    private static bool ContainsComment(string lineContent)
    {
        return lineContent.Contains("--") || lineContent.Contains("/*");
    }

    private ColumnDescription CreateColumnDescription(string tableName, SqlColumnDefinition column,
        List<ISqlExpression> allSqlExpressions)
    {
        return new ColumnDescription()
        {
            ColumnName = column.ColumnName,
            DataType = column.DataType,
            IsNullable = column.IsNullable,
            IsIdentity = IsIdentity(column.Identity),
            DefaultValue = column.Constraints.Where(x => x.SqlType == SqlType.ConstraintDefaultValue)
                .Cast<SqlConstraintDefaultValue>()
                .Select(x => x.DefaultValue)
                .FirstOrDefault(string.Empty),
            Description = allSqlExpressions
                .Where(x => x.SqlType == SqlType.AddExtendedProperty)
                .Cast<SqlSpAddExtendedProperty>()
                .Where(x => x.Name.Contains("MS_Description") && x.Level1Name == tableName &&
                            x.Level2Name == column.ColumnName)
                .Select(x => x.Value)
                .FirstOrDefault(string.Empty)
        };
    }

    private DatabaseDescription CreateDatabaseDescription(string databaseName, SqlFileContent sqlFileContent)
    {
        var database = new DatabaseDescription
        {
            DatabaseName = databaseName
        };
        var createTables = sqlFileContent.SqlExpressions
            .Where(x => x.SqlType == SqlType.CreateTable)
            .Cast<SqlCreateTableStatement>()
            .Where(x => StartsWithValidChar(x.TableName))
            .ToList();
        foreach (var createTable in createTables)
        {
            var table = CreateTableDescription(createTable, sqlFileContent.SqlExpressions);
            database.Tables.Add(table);
        }

        return database;
    }

    private static StreamWriter CreateStreamWriter(string createTablesFile)
    {
        var fileStream = new FileStream(createTablesFile, FileMode.Create);
        var writer = new StreamWriter(fileStream, Encoding.UTF8);
        return writer;
    }

    private TableDescription CreateTableDescription(SqlCreateTableStatement createTable,
        List<ISqlExpression> allSqlExpressions)
    {
        var tableName = createTable.TableName;
        var columns = createTable.Columns
            .Where(x => x.SqlType == SqlType.ColumnDefinition)
            .Cast<SqlColumnDefinition>()
            .ToList();
        var table = new TableDescription()
        {
            TableName = createTable.TableName,
            Columns = columns.Select(x => CreateColumnDescription(tableName, x, allSqlExpressions)).ToList()
        };
        return table;
    }

    private IEnumerable<DatabaseDescription> ExtractDatabaseDescriptions(IEnumerable<SqlFileContent> sqlContents)
    {
        foreach (var sqlFileContent in sqlContents)
        {
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            yield return CreateDatabaseDescription(databaseName, sqlFileContent);
        }
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

    private bool IsIdentity(SqlIdentity sqlIdentity)
    {
        return sqlIdentity.Increment > 0;
    }

    bool StartsWithValidChar(string text)
    {
        return !string.IsNullOrEmpty(text) && (char.IsLetter(text[0]) || text[0] == '_' || text[0] == '[');
    }
}

public class DatabaseDescription
{
    public string DatabaseName { get; set; } = string.Empty;
    public List<TableDescription> Tables { get; set; } = [];
}

public class TableDescription
{
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDescription> Columns { get; set; } = [];
}

public class ColumnDescription
{
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public bool IsNullable { get; set; }
    public bool IsIdentity { get; set; }
    public string DefaultValue { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
}

public class SqlFileContent
{
    public static SqlFileContent Empty => new();
    public string FileName { get; set; } = string.Empty;
    public string Sql { get; set; } = string.Empty;
    public List<ISqlExpression> SqlExpressions { get; set; } = [];
}

public class SqlCreateTablesSqlFiles
{
    public SqlFileContent File { get; set; } = SqlFileContent.Empty;
    public List<string> CreateTables { get; set; } = [];
    public string DatabaseName { get; set; } = string.Empty;
}