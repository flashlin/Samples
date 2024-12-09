using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using T1.SqlSharp.DatabaseDescriptions;
using T1.SqlSharp.Expressions;
using T1.Standard.Collections.Generics;
using T1.Standard.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace SqlSharpLit.Common.ParserLit;

public class ExtractSqlHelper
{
    private readonly IDatabaseNameProvider _databaseNameProvider;
    private readonly IJsonSerializer _jsonSerializer = new T1.Standard.Serialization.JsonSerializer();

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

    public void GenerateDatabasesDescriptionJonsFileFromFolder(string folder, string outputFolder)
    {
        if (!Directory.Exists(folder))
        {
            return;
        }

        var userDatabaseDesc = GetUserDatabaseDescription(outputFolder);

        var sqlFileContents = GetSqlContentsFromFolder(folder)
            .ToList();
        var databasesDesc = GetDatabaseDescriptions(sqlFileContents);


        foreach (var db in databasesDesc)
        {
            var tables = db.Tables
                .Where(x => !x.TableName.StartsWith("#"))
                .ToList();
            db.Tables = tables;
        }

        UpdateDatabaseDescription(databasesDesc, userDatabaseDesc);

        UpdateTableDescription(databasesDesc, userDatabaseDesc);

        SaveDatabasesDescJsonFile(databasesDesc, outputFolder);

        foreach (var database in databasesDesc)
        {
            WriteAllTableDescriptions(database, outputFolder);
        }
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
        foreach (var (sqlFile, selectSql) in ExtractStartSelectSqlString(folder))
        {
            ParseResult<SelectStatement> result;
            var sqlParser = new SqlParser(selectSql);
            try
            {
                result = sqlParser.ParseSelectStatement();
            }
            catch (Exception)
            {
                var sql = sqlParser.GetRemainingText();
                Console.WriteLine($"Error parsing position {sqlFile}:\n{sql}");
                throw;
            }

            if (!result.HasResult)
            {
                var msg = $"Error parsing {selectSql}\n{result.Error.Message}";
                throw new Exception(msg);
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
        }
    }

    private IEnumerable<(string FileName, string startSelectSql)> ExtractStartSelectSqlString(string folder)
    {
        foreach (var sqlFile in GetSqlTextFromFolder(folder))
        {
            foreach (var startSelectSql in ExtractSelectSqlFromText(sqlFile.Sql))
            {
                yield return (sqlFile.FileName, startSelectSql);
            }
        }
    }

    private IEnumerable<string> ExtractSelectSqlFromText(string text)
    {
        var select = "SELECT";
        var startSelectIndex = text.IndexOf(select, StringComparison.OrdinalIgnoreCase);
        if (startSelectIndex < 0)
        {
            yield break;
        }
        var startSelectSql = text.Substring(startSelectIndex);
        var nextChar = startSelectSql[select.Length];
        if (!char.IsWhiteSpace(nextChar))
        {
            var nextSelect = startSelectSql.Substring(select.Length);
            foreach (var subSelectSql in ExtractSelectSqlFromText(nextSelect))
            {
                yield return subSelectSql;
            }
            yield break;
        }
        yield return startSelectSql;
        foreach (var subSelectSql in ExtractSelectSqlFromText(startSelectSql.Substring(select.Length)))
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
        GenerateRagFilesFromSqlContents(sqlFileContents, outputFolder);
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
                .Cast<SqlSpAddExtendedPropertyExpression>()
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
            .Cast<SqlCreateTableExpression>()
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

    private TableDescription CreateTableDescription(SqlCreateTableExpression createTable,
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

    private static StreamWriter CreateWriter(string filename)
    {
        return new StreamWriter(Path.Combine("outputs", filename), false, Encoding.UTF8);
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

    private void GenerateDatabasesJsonFile(List<DatabaseDescription> databaseDescriptions, string outputJsonFile)
    {
        var json = JsonSerializer.Serialize(databaseDescriptions, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(outputJsonFile, json);
    }

    private List<DatabaseDescription> GetDatabaseDescriptions(List<SqlFileContent> sqlFileContents)
    {
        var databases = new EnsureKeyDictionary<string, DatabaseDescription>(databaseName => new DatabaseDescription()
        {
            DatabaseName = databaseName
        });
        foreach (var sqlFileContent in sqlFileContents)
        {
            Console.WriteLine($"Processing {sqlFileContent.FileName}");
            var databaseName = _databaseNameProvider.GetDatabaseNameFromPath(sqlFileContent.FileName);
            var db = databases[databaseName];
            var createTablesSql = sqlFileContent.SqlExpressions
                .Where(x => x.SqlType == SqlType.CreateTable)
                .Cast<SqlCreateTableExpression>()
                .OrderBy(x => x.TableName)
                .ToList();
            db.Tables.AddRange(createTablesSql.Select(x => CreateTableDescription(x, sqlFileContent.SqlExpressions)));
            var addExtendedProperties = sqlFileContent.SqlExpressions
                .Where(x => x.SqlType == SqlType.AddExtendedProperty)
                .Cast<SqlSpAddExtendedPropertyExpression>()
                .ToList();
            foreach (var extendedProperty in addExtendedProperties)
            {
                var tableName = extendedProperty.Level1Name;
                var table = db.Tables.FirstOrDefault(x => x.TableName == tableName);
                if (table == null)
                {
                    continue;
                }

                var columnName = extendedProperty.Level2Name;
                var column = table.Columns.First(x => x.ColumnName == columnName);
                column.Description = extendedProperty.Value;
            }

            databases[databaseName] = db;
        }

        return databases.Values.ToList();
    }

    private static List<DatabaseDescription> GetUserDatabaseDescription(string outputFolder)
    {
        var userDatabaseDescriptionYamlFile = Path.Combine(outputFolder, "DatabasesDescription.yaml");
        if (!File.Exists(userDatabaseDescriptionYamlFile))
        {
            return [];
        }

        var yamlSerializer = new YamlSerializer();
        var yaml = File.ReadAllText(userDatabaseDescriptionYamlFile);
        return yamlSerializer.Deserialize<List<DatabaseDescription>>(yaml);
    }

    private bool IsIdentity(SqlIdentity sqlIdentity)
    {
        return sqlIdentity.Increment > 0;
    }

    private void SaveDatabasesDescJsonFile(List<DatabaseDescription> databasesDesc, string outputFolder)
    {
        var json = _jsonSerializer.Serialize(databasesDesc);
        using var writer = CreateStreamWriter(Path.Combine(outputFolder, "DatabasesDescription.json"));
        writer.Write(json);
        writer.Flush();
    }

    bool StartsWithValidChar(string text)
    {
        return !string.IsNullOrEmpty(text) && (char.IsLetter(text[0]) || text[0] == '_' || text[0] == '[');
    }

    private static void UpdateDatabaseDescription(List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        var innerDatabases = databasesDesc.Join(userDatabaseDesc,
                db => db.DatabaseName,
                udb => udb.DatabaseName,
                (db, udb) => new { Database = db, UserDatabase = udb })
            .ToList();
        foreach (var desc in innerDatabases)
        {
            desc.Database.Description = desc.UserDatabase.Description;
        }
    }

    private static void UpdateTableDescription(List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        var tables = databasesDesc.SelectMany(x => x.Tables, (db, table) => new
        {
            db.DatabaseName,
            Table = table
        }).ToList();
        var userTables = userDatabaseDesc.SelectMany(x => x.Tables, (db, table) => new
        {
            DatabaseName = db.DatabaseName,
            Table = table
        }).ToList();
        var innerTables = tables.Join(userTables,
                t => new { t.DatabaseName, t.Table.TableName },
                ut => new { ut.DatabaseName, ut.Table.TableName },
                (t, ut) => new { Table = t.Table, UserTable = ut.Table })
            .ToList();
        foreach (var tableDesc in innerTables)
        {
            tableDesc.Table.Description = tableDesc.UserTable.Description;

            var innerColumns = tableDesc.Table.Columns.Join(tableDesc.UserTable.Columns,
                    c => c.ColumnName,
                    uc => uc.ColumnName,
                    (c, uc) => new { Column = c, UserColumn = uc })
                .ToList();
            foreach (var columnDesc in innerColumns)
            {
                columnDesc.Column.Description = columnDesc.UserColumn.Description;
            }
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