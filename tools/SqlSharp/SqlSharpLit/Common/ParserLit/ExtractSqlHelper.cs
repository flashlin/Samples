using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
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
    private readonly IDatabaseNameProvider _databaseNameProvider;
    private readonly IJsonSerializer _jsonSerializer = new T1.Standard.Serialization.JsonSerializer();
    private const string DatabasesDescriptionName = "DatabasesDescription";

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
    
    public void GenerateDatabasesDescriptionFileFromFolder(string createTablesSqlFolder, string outputFolder)
    {
        if (!Directory.Exists(createTablesSqlFolder))
        {
            return;
        }
        var outputFile = Path.Combine(outputFolder, $"{DatabasesDescriptionName}_FromSqlFiles.json");
        if(File.Exists(outputFile))
        {
            return;
        }
        var databases = ExtractDatabasesDescriptionFromFolder(createTablesSqlFolder);
        SaveDatabasesDescriptionJsonFile(databases, outputFile);
    }
    
    public void MergeUserDatabasesDescription(string outputFolder)
    {
        var userDatabaseDescriptionYamlFile = Path.Combine(outputFolder, $"../{DatabasesDescriptionName}.yaml");
        var userDatabase = GetUserDatabaseDescription(userDatabaseDescriptionYamlFile);
        
        var databasesDescriptionFromSqlFilesOfJsonFile = Path.Combine(outputFolder, $"{DatabasesDescriptionName}_FromSqlFiles.json");
        var databases = LoadDatabasesDescriptionJsonFile(databasesDescriptionFromSqlFilesOfJsonFile);

        var updatedDatabases = databases.UpdateDatabaseDescription(userDatabase);
        UpdateTableDescription(updatedDatabases, userDatabase);
        var databasesDescriptionFinishFile = Path.Combine(outputFolder, $"{DatabasesDescriptionName}_User.json");
        SaveDatabasesDescriptionJsonFile(updatedDatabases, databasesDescriptionFinishFile);
    }

    private static List<DatabaseDescription> LoadDatabasesDescriptionJsonFile(string databasesDescriptionFromSqlFilesOfJsonFile)
    {
        var json = File.ReadAllText(databasesDescriptionFromSqlFilesOfJsonFile);
        var databases = JsonSerializer.Deserialize<List<DatabaseDescription>>(json)!;
        return databases;
    }
    
    public void GenerateDatabaseDescriptionsQaMdFile(string databaseDescriptionsFile)
    {
        var outputFolder = Path.GetDirectoryName(databaseDescriptionsFile)!;
        using var databaseSchemaQaWriter = new DatabaseSchemaQaWriter(outputFolder);
        var databases = LoadDatabasesDescriptionJsonFile(databaseDescriptionsFile);
        databaseSchemaQaWriter.GenerateQaMdFile(databases);
    }

    public void GenerateDatabasesDescriptionJonsFileFromFolder(string createTablesSqlFolder, string outputFolder)
    {
        if (!Directory.Exists(createTablesSqlFolder))
        {
            return;
        }

        var outputParentFolder = Path.GetDirectoryName(outputFolder)!;
        var userDatabase = GetUserDatabaseDescription(Path.Combine(outputParentFolder, "DatabasesDescription.yaml"));
        var databases = ExtractDatabasesDescriptionFromFolder(createTablesSqlFolder);

        var updatedDatabases = databases.UpdateDatabaseDescription(userDatabase);
        UpdateTableDescription(updatedDatabases, userDatabase);
        SaveDatabasesDescriptionJsonFile(updatedDatabases, Path.Combine(outputFolder, "DatabasesDescription.json"));

        using var databaseSchemaQaWriter = new DatabaseSchemaQaWriter(outputFolder);
        databaseSchemaQaWriter.GenerateQaMdFile(updatedDatabases);
    }

    private List<DatabaseDescription> ExtractDatabasesDescriptionFromFolder(string folder)
    {
        var sqlFileContents = GetSqlContentsFromFolder(folder)
            .ToList();
        var databasesDesc = CreateDatabaseDescriptions(sqlFileContents);
        NormalizeDatabaseDescriptions(databasesDesc);
        foreach (var db in databasesDesc)
        {
            var tables = db.Tables
                .Where(x => !x.TableName.StartsWith("#"))
                .ToList();
            db.Tables = tables;
        }
        return databasesDesc;
    }

    private void NormalizeDatabaseDescriptions(List<DatabaseDescription> databasesDesc)
    {
        foreach (var database in databasesDesc)
        {
            foreach (var table in database.Tables)
            {
                var tableName = NormalizeName(table.TableName);
                table.TableName = tableName;
                foreach (var column in table.Columns)
                {
                    column.ColumnName = NormalizeName(column.ColumnName);
                }
            }
        }
    }

    private static string NormalizeName(string tableName)
    {
        tableName = tableName.Replace("[dbo].", "");
        tableName = Regex.Replace(tableName, @"\[(.*?)\]", "$1");
        return tableName;
    }

    public void GenerateRagFiles(string sqlFolder)
    {
        var sqlFileContents = GetSqlContentsFromFolder(sqlFolder);
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

    public void ExtractSelectSqlFromFolder(string folder, string outputFile)
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

    private IEnumerable<SqlSelectContent> ExtractSelectStatement(string folder)
    {
        foreach (var (sqlFile, selectSql) in ExtractStartSelectSqlString(folder))
        {
            var sqlParser = new SqlParser(selectSql);
            var result = sqlParser.ParseSelectStatement();
            if (result.HasError)
            {
                throw result.Error;
            }

            yield return new SqlSelectContent
            {
                FileName = sqlFile,
                Statement = result.ResultValue
            };
        }
    }

    public void GenerateSelectStatementQaMdFile(string folder, string outputFile)
    {
        var selectContents = ExtractSelectStatement(folder);
        using var writer = new StreamWriter(outputFile, true, Encoding.UTF8);
        foreach (var selectContent in selectContents)
        {
            Console.WriteLine($"Processing {selectContent.FileName}");
            writer.WriteLine($"## {selectContent.FileName}");
            writer.WriteLine(selectContent.Statement.ToSql());
            writer.WriteLine();
            writer.Flush();
        }
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

    private string GetErrorMessage(string sqlFile, SqlParser sqlParser)
    {
        var msg = $"{sqlFile}:\n";
        var positionSql = sqlParser.GetRemainingText();
        msg += $"GetRemainingText:\n{positionSql}\n";
        msg += $"----------\n{sqlParser.GetPreviousText(0)}";
        return msg;
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

    private IEnumerable<(string FileName, string startSelectSql)> ExtractStartSelectSqlString(string folder)
    {
        foreach (var sqlFile in GetSqlTextFromFolder(folder))
        {
            var sql = ExcludeSqlComments(sqlFile.Sql);
            sql = ExcludeNonSelectSql(sql);
            foreach (var startSelectSql in ExtractSelectSqlFromText(sql))
            {
                yield return (sqlFile.FileName, startSelectSql);
            }
        }
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

    private bool IsCommentLine(string line)
    {
        return line.StartsWith("/*") || line.StartsWith("--");
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

    private IEnumerable<string> ExtractSelectSqlFromText(string text, int startOffset = 0)
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
            foreach (var subSelectSql in ExtractSelectSqlFromText(text, startOffset + select.Length))
            {
                yield return subSelectSql;
            }

            yield break;
        }

        yield return startSelectSql;
        foreach (var subSelectSql in ExtractSelectSqlFromText(text, startOffset + startSelectSql.Length))
        {
            yield return subSelectSql;
        }
    }

    public IEnumerable<SqlFileContent> GetSqlTextFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            var sql = File.ReadAllText(sqlFile);
            yield return new SqlFileContent
            {
                FileName = sqlFile,
                Sql = sql,
            };
        }
    }

    public IEnumerable<SqlFileContent> GetSqlContentsFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            Console.WriteLine($"Parsing {sqlFile}");
            var sql = File.ReadAllText(sqlFile);
            List<ISqlExpression> sqlExpressions;
            try
            {
                sqlExpressions = new SqlParser(sql).Extract().ToList();
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error parsing {sqlFile}");
                Console.WriteLine(e.Message);
                continue;
            }

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
        if (!Directory.Exists(folder))
        {
            yield break;
        }

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

    public void WriteCreateTablesFromFolder(string createTablesSqlFolder, string outputFolder)
    {
        if (!Directory.Exists(createTablesSqlFolder))
        {
            return;
        }
        using var writer = StreamWriterCreator.Create(Path.Combine(outputFolder, "CreateTables.sql"));
        var sqlFileContents = GetSqlContentsFromFolder(createTablesSqlFolder)
            .ToList();
        WriteCreateTablesTo(sqlFileContents, writer);
        GenerateRagFilesFromSqlContents(sqlFileContents, outputFolder);
    }

    private static bool ContainsComment(string lineContent)
    {
        return lineContent.Contains("--") || lineContent.Contains("/*");
    }

    private static StreamWriter CreateWriter(string filename)
    {
        return new StreamWriter(Path.Combine("outputs", filename), false, Encoding.UTF8);
    }

    private IEnumerable<DatabaseDescription> ExtractDatabaseDescriptions(IEnumerable<SqlFileContent> sqlContents)
    {
        foreach (var sqlFileContent in sqlContents)
        {
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            yield return DatabaseDescriptionCreator.CreateDatabaseDescription(databaseName, sqlFileContent.SqlExpressions);
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

    private void GenerateDatabasesJsonFile(List<DatabaseDescription> databaseDescriptions, string outputJsonFile)
    {
        var json = JsonSerializer.Serialize(databaseDescriptions, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(outputJsonFile, json);
    }

    private List<DatabaseDescription> CreateDatabaseDescriptions(List<SqlFileContent> sqlFileContents)
    {
        var databases = new EnsureKeyDictionary<string, DatabaseDescription>(databaseName => new DatabaseDescription()
        {
            DatabaseName = databaseName
        });
        foreach (var sqlFileContent in sqlFileContents)
        {
            Console.WriteLine($"Processing {sqlFileContent.FileName}");
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            var newDb = DatabaseDescriptionCreator.CreateDatabaseDescription(databaseName, sqlFileContent.SqlExpressions);
            
            var db = databases[databaseName];
            db.Tables.AddRange(newDb.Tables);
            databases[databaseName] = db;
        }
        return databases.Values.ToList();
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

    private void SaveDatabasesDescriptionJsonFile(List<DatabaseDescription> databasesDesc, string outputFile)
    {
        var json = _jsonSerializer.Serialize(databasesDesc);
        using var writer = StreamWriterCreator.Create(outputFile);
        writer.Write(json);
        writer.Flush();
    }

    private static void UpdateTableDescription(List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        foreach (var database in databasesDesc)
        {
            var tables = database.Tables;
            var userTables = userDatabaseDesc.FirstOrDefault(x => x.DatabaseName.IsSameAs(database.DatabaseName))?.Tables ??
                             [];
            var updatedTables = tables.LeftOuterJoin(userTables,
                    ut => ut.TableName,
                    t => t.TableName,
                    ut => ut,
                    (t, ut) => new TableDescription()
                    {
                        TableName = t.TableName,
                        Description = string.IsNullOrEmpty(ut.Description) ? t.Description : ut.Description,
                        Columns = t.Columns
                    })
                .ToList();
            database.Tables = updatedTables;
            tables.UpdateTableColumnsDescription(userTables);
        }
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

    public void SetDatabaseNameDeep(int deep)
    {
        _databaseNameProvider.SetDeep(deep);
    }
}

public class SqlSelectContent
{
    public string FileName { get; set; } = string.Empty;
    public required SelectStatement Statement { get; set; }
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