using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using T1.SqlSharp;
using T1.SqlSharp.DatabaseDescriptions;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;
using T1.SqlSharp.Helper;
using T1.Standard.Collections.Generics;
using T1.Standard.Linq;
using T1.Standard.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace SqlSharpLit.Common.ParserLit;

public class ExtractSqlHelper
{
    private const string DatabasesDescriptionName = "DatabasesDescription";
    private readonly IDatabaseNameProvider _databaseNameProvider;
    private readonly ExtractSqlFileHelper _extractSqlFileHelper;
    private readonly JsonDocSerializer _jsonDocSerializer = new();

    public ExtractSqlHelper(IDatabaseNameProvider databaseNameProvider)
    {
        _databaseNameProvider = databaseNameProvider;
        _extractSqlFileHelper = new ExtractSqlFileHelper();
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

    public void TestExtractSelectSqlFromFolder(string folder, string outputFile)
    {
        if (File.Exists(outputFile))
        {
            File.Delete(outputFile);
        }

        var fileCount = 0;
        foreach (var (sqlFile, selectSql) in ExtractStartSelectSqlString(folder))
        {
            Console.WriteLine($"Processing {sqlFile}");
            ParseResult<SelectStatement> result;
            var sqlParser = new SqlParser(selectSql);
            try
            {
                result = sqlParser.ParseSelectStatement();
            }
            catch (Exception)
            {
                var remainingSql = sqlParser.GetRemainingText();
                Console.Clear();
                Console.WriteLine("Processed files count: " + fileCount);
                Console.WriteLine($"Exception {sqlFile}:\n{remainingSql}");
                Console.WriteLine($"----------\n{sqlParser.GetPreviousText(0)}");
                throw;
            }

            if (result.HasError)
            {
                WriteErrorSqlFile(sqlFile, sqlParser);
                var msg = result.Error.Message + "\n" + GetErrorMessage(sqlFile, sqlParser);
                throw new Exception(msg);
            }

            if (!result.HasResult)
            {
                WriteErrorSqlFile(sqlFile, sqlParser);
                throw new Exception("No result");
            }

            try
            {
                File.AppendAllText(outputFile, result.ResultValue.ToSql());
            }
            catch (Exception)
            {
                Console.WriteLine($"Error ToSql:\n {JsonSerializer.Serialize(result.ResultValue)}");
                var sql = sqlParser.GetRemainingText();
                Console.WriteLine($"Error file {sqlFile}:\n{sql}");
                throw;
            }

            fileCount++;
        }
    }

    public static bool FindOccurrences(string text, int n)
    {
        if (string.IsNullOrEmpty(text) || n <= 0)
        {
            return false;
        }

        var count = 0;
        for (var i = 0; i < text.Length - 1; i++)
        {
            if (text[i] == '\'' && text[i + 1] == '\'')
            {
                count++;
                i++;
                if (count == n)
                {
                    return true;
                }
            }
        }

        return false;
    }

    public void GenerateDatabaseDescriptionsQaMdFile(string databaseDescriptionsFile)
    {
        var outputFolder = Path.GetDirectoryName(databaseDescriptionsFile)!;
        using var databaseSchemaQaWriter = new DatabaseSchemaQaWriter(outputFolder);
        var databases = LoadDatabasesDescriptionJsonFile(databaseDescriptionsFile);
        databaseSchemaQaWriter.GenerateQaMdFile(databases);
    }

    public void GenerateDatabasesDescriptionJsonFileFromFolder(string createTablesSqlFolder, string outputFolder)
    {
        if (!Directory.Exists(createTablesSqlFolder))
        {
            return;
        }

        var databases = ExtractDatabasesDescriptionFromFolder(createTablesSqlFolder);
        var outputFile = Path.Combine(outputFolder, $"{DatabasesDescriptionName}_FromSqlFiles.json");
        SaveDatabasesDescriptionJsonFile(databases, outputFile);
    }

    public void GenerateRagFiles(string sqlFolder)
    {
        var sqlFileContents = _extractSqlFileHelper.GetSqlContentsFromFolder(sqlFolder);
        GenerateRagFilesFromSqlContents(sqlFileContents, "outputs");
    }

    public void GenerateRagFilesFromSqlContents(IEnumerable<SqlFileContent> sqlFileContents, string outputsFolder)
    {
        var databaseDescriptions = ExtractDatabaseDescriptions(sqlFileContents)
            .Where(x => x.Tables.Count > 0)
            .ToList();

        databaseDescriptions = databaseDescriptions.GroupBy(x => x.DatabaseName)
            .Select(x => new DatabaseDescription()
            {
                DatabaseName = x.Key,
                Tables = x.SelectMany(y => y.Tables).ToList()
            })
            .OrderBy(x => x.DatabaseName)
            .ToList();


        GenerateDatabasesJsonFile(databaseDescriptions, Path.Combine("outputs", "Databases.json"));
        //writer.WriteLine("The following is a detailed description of all databases and table structures of Titan Company.");
        foreach (var database in databaseDescriptions)
        {
            WriteAllTableDescriptions(database, outputsFolder);
            WriteAllDatabaseTableNames(database);
        }

        WriteDatabaseNamesDesc(databaseDescriptions);
        WriteDatabaseTableNamesDesc(databaseDescriptions);
    }

    public async Task GenerateSelectStatementQaMdFileAsync(string folder, string outputFolder)
    {
        if (!Directory.Exists(folder))
        {
            return;
        }
        var databasesDescription =
            LoadDatabasesDescriptionJsonFile(Path.Combine(outputFolder, $"{DatabasesDescriptionName}_User.json"));
        
        var selectSqlList = ExtractSelectSql(folder, databasesDescription).ToList();
        
        await using var mdWriter = StreamWriterCreator.Create(Path.Combine(outputFolder, "SelectQa.md"));
        var count = 0;
        foreach (var item in selectSqlList)
        {
            await mdWriter.WriteLineAsync($"# Database: {item.Database.DatabaseName}");
            var tableNames = string.Join(",", item.Database.Tables.Select(x=>x.TableName).ToList());
            await mdWriter.WriteLineAsync($"## Tables: {tableNames}");
            var tableSources = item.SelectSql.FromSources
                .Where(x => x.SqlType == SqlType.TableSource)
                .Cast<SqlTableSource>()
                .ToList();
            var tableSourceNames = string.Join(",", tableSources.Select(x=>x.TableName).ToList());
            await mdWriter.WriteLineAsync($"## Table Sources: {tableSourceNames}");
            await mdWriter.WriteLineAsync($"{item.SelectSql.ToSql()}");
            count++;
        }
        await mdWriter.WriteLineAsync($"Total: {count}");

        count = 0;
        using var csv = new CsvSharpWriter();
        await csv.CreateFileAsync<CsvSelectQaPrompt>(Path.Combine(outputFolder, "SelectQaPrompt.csv"));
        foreach (var prompt in GenerateSelectSqlPrompt(selectSqlList))
        {
            await csv.WriteRecordAsync(new CsvSelectQaPrompt
            {
                Prompt = prompt
            });
            count++;
        }
        await csv.FlushAsync();
        csv.Close();
        Console.WriteLine($"Total: {count}");

        using var csvReader = new CsvSharpReader();
        await csvReader.OpenFileAsync<CsvSelectQaPrompt>(Path.Combine(outputFolder, "SelectQaPrompt.csv"));
        count = 0;
        await foreach (var record in csvReader.ReadRecordsAsync<CsvSelectQaPrompt>(x=>new CsvSelectQaPrompt{Prompt = x.Prompt}))
        {
            count++;
        }
        Console.WriteLine($"CsvTotal: {count}");
    }


    private IEnumerable<DatabaseSelectSql> ExtractSelectSql(string folder, List<DatabaseDescription> databasesDescription)
    {
        foreach (var selectContent in ExtractSelectStatement(folder))
        {
            Console.WriteLine($"Processing {selectContent.FileName}");
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(selectContent.FileName);
            var db = databasesDescription.FirstOrDefault(x => x.DatabaseName.IsNormalizeSameAs(databaseName));
            if (db == null)
            {
                throw new KeyNotFoundException($"{selectContent.FileName} Db:{databaseName}");
                continue;
            }
            var selectFromTableSourceStatements = ExtractSelectFromTableSourceSql(selectContent.Statements)
                .ToList();
            foreach (var selectFromTableSourceStatement in selectFromTableSourceStatements)
            {
                yield return new DatabaseSelectSql
                {
                    Database = db, 
                    SelectSql = selectFromTableSourceStatement
                };
            }
        }
    }

    private IEnumerable<string> GenerateSelectSqlPrompt(IEnumerable<DatabaseSelectSql> selectSqlList)
    {
        foreach (var item in selectSqlList)
        {
            var databaseName = item.Database.DatabaseName;
            var selectFromTableSourceStatement = item.SelectSql;
            var writer = new StringWriter();
            writer.WriteLine($"## Database: {databaseName}");
            var tableSources = selectFromTableSourceStatement.FromSources
                .Where(x => x.SqlType == SqlType.TableSource)
                .Cast<SqlTableSource>()
                .ToList();
            var tableNames = tableSources.Select(x => x.TableName.NormalizeName()).ToList();
            var foundTables = 0;
            foreach (var tableName in tableNames)
            {
                var table = item.Database.Tables.FirstOrDefault(x => x.TableName.IsNormalizeSameAs(tableName));
                if (table == null)
                {
                    continue;
                }
                foundTables++;
                writer.WriteLine($"### {tableName}");
                writer.WriteLine(table.ToDescriptionText());
            }
            if (foundTables == 0)
            {
                continue;
            }

            writer.WriteLine();
            writer.WriteLine("以上是關於 table 的描述");
            writer.WriteLine(
                "以下 SQL 內容是 AI 根據使用者的回答生成的. 請反推出使用者當時是詢問什麼商業業務問題? 用一條疑問句就好, 不要有技術性的內容\n例如: 取得用戶調查的基本資訊以及他們對特定問題的回答");
            writer.WriteLine("```sql");
            writer.WriteLine(selectFromTableSourceStatement.ToSql());
            writer.WriteLine("```");
            writer.WriteLine();
            yield return writer.ToString();
        }
    }

    private IEnumerable<SelectStatement> ExtractSelectFromTableSourceSql(IEnumerable<SelectStatement> selectStatements)
    {
        foreach (var selectStatement in selectStatements)
        {
            var tableSources = selectStatement.FromSources
                .Where(x => x.SqlType == SqlType.TableSource)
                .Cast<SqlTableSource>()
                .ToList();
            if (tableSources.Count == 0)
            {
                continue;
            }
            foreach (var tableSource in tableSources)
            {
                tableSource.TableName = tableSource.TableName.NormalizeName();
            }
            yield return selectStatement;
        }
    }

    public IEnumerable<SqlCreateTablesSqlFiles> GetCreateCreateTableSqlFromFolder(
        IEnumerable<SqlFileContent> sqlContents)
    {
        foreach (var sqlFileContent in sqlContents)
        {
            Console.WriteLine($"{sqlFileContent.FileName}");
            var createTablesSql = ExtractAllCreateTableFromText(sqlFileContent.Sql).ToList();
            yield return new SqlCreateTablesSqlFiles
            {
                File = sqlFileContent,
                DatabaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName),
                CreateTables = createTablesSql,
            };
        }
    }

    public void MergeUserDatabasesDescription(string outputFolder)
    {
        var userDatabaseDescriptionYamlFile = Path.Combine(outputFolder, $"../{DatabasesDescriptionName}.yaml");
        var userDatabase = GetUserDatabaseDescription(userDatabaseDescriptionYamlFile);

        var databasesDescriptionFromSqlFilesJsonFile =
            Path.Combine(outputFolder, $"{DatabasesDescriptionName}_FromSqlFiles.json");
        if (!File.Exists(databasesDescriptionFromSqlFilesJsonFile))
        {
            return;
        }

        var databases = LoadDatabasesDescriptionJsonFile(databasesDescriptionFromSqlFilesJsonFile);

        databases.UpdateDatabaseDescription(userDatabase);
        SaveDatabasesDescriptionJsonFile(databases, databasesDescriptionFromSqlFilesJsonFile);
    }

    public void SetDatabaseNameDeep(int deep)
    {
        Console.WriteLine($"set deep: {deep}");
        _databaseNameProvider.SetDeep(deep);
    }

    public void WriteCreateTablesFromFolder(string createTablesSqlFolder, string outputFolder)
    {
        if (!Directory.Exists(createTablesSqlFolder))
        {
            return;
        }

        using var writer = StreamWriterCreator.Create(Path.Combine(outputFolder, "CreateTables.sql"));
        var sqlFileContents = _extractSqlFileHelper.GetSqlContentsFromFolder(createTablesSqlFolder)
            .ToList();
        WriteCreateTablesTo(sqlFileContents, writer);
        GenerateRagFilesFromSqlContents(sqlFileContents, outputFolder);
    }

    private static bool ContainsComment(string lineContent)
    {
        return lineContent.Contains("--") || lineContent.Contains("/*");
    }

    private List<DatabaseDescription> CreateDatabasesDescription(List<SqlFileContent> sqlFileContents)
    {
        var databases = new EnsureKeyDictionary<string, DatabaseDescription>(databaseName => new DatabaseDescription()
        {
            DatabaseName = databaseName
        });
        foreach (var sqlFileContent in sqlFileContents)
        {
            Console.WriteLine($"Processing {sqlFileContent.FileName}");
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            var newDb = DatabaseDescriptionCreator.CreateDatabaseDescription(databaseName,
                sqlFileContent.SqlExpressions);

            var db = databases[databaseName];
            db.Tables.AddRange(newDb.Tables);
            databases[databaseName] = db;
        }

        return databases.Values.ToList();
    }

    private static StreamWriter CreateWriter(string filename)
    {
        return new StreamWriter(Path.Combine("outputs", filename), false, Encoding.UTF8);
    }

    private string ExcludeNonSelectSql(string text)
    {
        if (text.Contains("'''"))
        {
            return string.Empty;
        }

        if (FindOccurrences(text, 2))
        {
            return string.Empty;
        }

        if (text.Contains("set @sql ="))
        {
            return string.Empty;
        }

        return text;
    }


    private string ExcludeSqlComments(string text)
    {
        var lines = text.Split(["\r\n", "\n"], StringSplitOptions.None);
        var result = new StringBuilder();
        var inComment = false;
        foreach (var line in lines)
        {
            if (line.Trim().StartsWith("/*") && line.Trim().EndsWith("*/"))
            {
                continue;
            }

            if (line.TrimStart().StartsWith("/*"))
            {
                inComment = true;
                continue;
            }

            if (line.TrimEnd().EndsWith("*/"))
            {
                inComment = false;
                continue;
            }

            if (inComment)
            {
                continue;
            }

            if (line.TrimStart().StartsWith("--"))
            {
                continue;
            }

            result.AppendLine(line);
        }

        return result.ToString();
    }

    private IEnumerable<DatabaseDescription> ExtractDatabaseDescriptions(IEnumerable<SqlFileContent> sqlContents)
    {
        foreach (var sqlFileContent in sqlContents)
        {
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            yield return DatabaseDescriptionCreator.CreateDatabaseDescription(databaseName,
                sqlFileContent.SqlExpressions);
        }
    }

    private List<DatabaseDescription> ExtractDatabasesDescriptionFromFolder(string folder)
    {
        var sqlFileContents = _extractSqlFileHelper.GetSqlContentsFromFolder(folder)
            .ToList();
        var databasesDesc = CreateDatabasesDescription(sqlFileContents);
        foreach (var db in databasesDesc)
        {
            var tables = db.Tables
                .Where(x => !x.TableName.StartsWith("#"))
                .ToList();
            db.Tables = tables;
        }

        return databasesDesc;
    }

    private IEnumerable<string> ExtractStartSelectSqlTextFromText(string text, int startOffset = 0)
    {
        var select = "SELECT";
        var startSelectIndex = text.IndexOf(select, startOffset, StringComparison.OrdinalIgnoreCase);
        if (startSelectIndex < 0)
        {
            yield break;
        }

        var startSelectSql = text.Substring(startSelectIndex);
        var nextChar = startSelectSql[select.Length];
        if (!char.IsWhiteSpace(nextChar))
        {
            foreach (var subSelectSql in ExtractStartSelectSqlTextFromText(text, startOffset + select.Length))
            {
                yield return subSelectSql;
            }

            yield break;
        }

        yield return startSelectSql;
        foreach (var subSelectSql in ExtractStartSelectSqlTextFromText(text, startOffset + startSelectSql.Length))
        {
            yield return subSelectSql;
        }
    }

    private IEnumerable<SqlSelectContent> ExtractSelectStatement(string folder)
    {
        var currentSqlFile = SqlSelectContent.Empty;
        foreach (var (sqlFile, selectSql) in ExtractStartSelectSqlString(folder))
        {
            var sqlParser = new SqlParser(selectSql);
            ParseResult<SelectStatement> result;
            try
            {
                result = sqlParser.ParseSelectStatement();
            }
            catch (Exception)
            {
                var remainingSql = sqlParser.GetRemainingText();
                Console.WriteLine($"Exception {sqlFile}:\n{remainingSql}");
                Console.WriteLine($"----------\n{sqlParser.GetPreviousText(0)}");
                continue;
            }

            if (result.HasError)
            {
                continue;
            }

            if (currentSqlFile.FileName != sqlFile)
            {
                if (currentSqlFile.HasSelectSql())
                {
                    yield return currentSqlFile;
                }

                currentSqlFile = new SqlSelectContent
                {
                    FileName = sqlFile
                };
            }

            currentSqlFile.Statements.Add(result.ResultValue);
        }

        if (currentSqlFile.HasSelectSql())
        {
            yield return currentSqlFile;
        }
    }

    private IEnumerable<(string FileName, string startSelectSql)> ExtractStartSelectSqlString(string folder)
    {
        foreach (var sqlFile in _extractSqlFileHelper.GetSqlTextFromFolder(folder))
        {
            var sql = ExcludeSqlComments(sqlFile.Sql);
            sql = ExcludeNonSelectSql(sql);
            foreach (var startSelectSql in ExtractStartSelectSqlTextFromText(sql))
            {
                yield return (sqlFile.FileName, startSelectSql);
            }
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

    private (string truncatedText, int length) FindCreateTableStart(string text)
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

    private void GenerateDatabasesJsonFile(List<DatabaseDescription> databaseDescriptions, string outputJsonFile)
    {
        var json = JsonSerializer.Serialize(databaseDescriptions, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(outputJsonFile, json);
    }

    private string GetErrorMessage(string sqlFile, SqlParser sqlParser)
    {
        var msg = $"{sqlFile}:\n";
        var positionSql = sqlParser.GetRemainingText();
        msg += $"GetRemainingText:\n{positionSql}\n";
        msg += $"----------\n{sqlParser.GetPreviousText(0)}";
        return msg;
    }

    private string GetPreviousLine(string text, int offset)
    {
        var lastNewLineIndex = text.LastIndexOf('\n', offset - 1);
        if (lastNewLineIndex < 0)
        {
            return string.Empty;
        }

        var line = text.Substring(lastNewLineIndex + 1, offset - lastNewLineIndex - 1);
        return line;
    }

    private static List<DatabaseDescription> GetUserDatabaseDescription(string userDatabaseDescriptionYamlFile)
    {
        if (!File.Exists(userDatabaseDescriptionYamlFile))
        {
            return [];
        }

        var yamlSerializer = new YamlSerializer();
        var yaml = File.ReadAllText(userDatabaseDescriptionYamlFile);
        return yamlSerializer.Deserialize<List<DatabaseDescription>>(yaml);
    }

    private bool IsCommentLine(string line)
    {
        return line.StartsWith("/*") || line.StartsWith("--");
    }

    private static List<DatabaseDescription> LoadDatabasesDescriptionJsonFile(
        string databasesDescriptionFromSqlFilesOfJsonFile)
    {
        var json = File.ReadAllText(databasesDescriptionFromSqlFilesOfJsonFile);
        var databases = JsonSerializer.Deserialize<List<DatabaseDescription>>(json)!;
        return databases;
    }

    private void SaveDatabasesDescriptionJsonFile(List<DatabaseDescription> databasesDesc, string outputFile)
    {
        _jsonDocSerializer.WriteToJsonFile(databasesDesc, outputFile);
    }

    private static void WriteAllDatabaseTableNames(DatabaseDescription database)
    {
        var databaseDescriptionMdFile = Path.Combine("outputs", $"Database-Tables-{database.DatabaseName}-Desc.txt");
        using var writer = new StreamWriter(databaseDescriptionMdFile, false, Encoding.UTF8);
        WriteDatabaseTableNamesTo(database, writer);
        writer.Flush();
    }

    private static void WriteAllTableDescriptions(DatabaseDescription database, string outputFolder)
    {
        var databaseDescriptionMdFile = Path.Combine(outputFolder, $"Database-{database.DatabaseName}-Desc.txt");
        using var writer = new StreamWriter(databaseDescriptionMdFile, false, Encoding.UTF8);
        foreach (var table in database.Tables)
        {
            WriteTableDescription(writer, database, table);
        }

        writer.Flush();
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

    private void WriteDatabaseNamesDesc(List<DatabaseDescription> databaseDescriptions)
    {
        using var writer = CreateWriter("Database-Names-Desc.txt");
        writer.WriteLine("The following is a list of database lists:");
        foreach (var database in databaseDescriptions)
        {
            writer.WriteLine($"{database.DatabaseName}");
        }

        writer.Flush();
    }

    private static void WriteDatabaseTableNamesDesc(List<DatabaseDescription> databaseDescriptions)
    {
        using var writer = CreateWriter("Database-TableNames-Desc.txt");
        foreach (var database in databaseDescriptions)
        {
            WriteDatabaseTableNamesTo(database, writer);
        }

        writer.Flush();
    }

    private static void WriteDatabaseTableNamesTo(DatabaseDescription database, StreamWriter writer)
    {
        writer.WriteLine($"Database Name: {database.DatabaseName}");
        writer.WriteLine("Tables:");
        foreach (var table in database.Tables)
        {
            writer.WriteLine($"{table.TableName}");
        }

        writer.WriteLine();
        writer.WriteLine();
        writer.WriteLine();
    }

    private void WriteErrorSqlFile(string sqlFile, SqlParser sqlParser)
    {
        var errorFile = Path.Combine("outputs", "error.sql");
        var sql = sqlParser.GetPreviousText(0);
        var msg = $"{sqlFile}\n";
        msg += sql + "\n";
        msg += "----------------\n";
        msg += sqlParser.GetRemainingText();
        File.WriteAllText(errorFile, msg);
    }

    private static void WriteTableDescription(StreamWriter writer,
        DatabaseDescription database, TableDescription table)
    {
        writer.Write($"Database Name: {database.DatabaseName}");
        if (!string.IsNullOrEmpty(database.Description))
        {
            writer.Write($" -- {database.Description}");
        }

        writer.WriteLine();
        writer.Write($"Table Name: {table.TableName}");
        if (!string.IsNullOrEmpty(table.Description))
        {
            writer.Write($" -- {table.Description}");
        }

        writer.WriteLine();
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
                writer.Write($" -- {column.Description}");
            }

            writer.WriteLine();
        }

        writer.WriteLine();
        writer.WriteLine();
    }
}

public class DatabaseSelectSql
{
    public required DatabaseDescription Database { get; set; }
    public required SelectStatement SelectSql { get; set; }
}

public class CsvSelectQaPrompt
{
    public string Prompt { get; set; } = string.Empty;
}