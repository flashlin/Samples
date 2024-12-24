using System.Text.Json;
using T1.SqlSharp.DatabaseDescriptions;

namespace SqlSharpLit.Common.ParserLit;

public class DatabaseSchemaQaWriter : IDisposable
{
    private readonly StreamWriter _writer;

    public DatabaseSchemaQaWriter(string outputFolder)
    {
        _writer = StreamWriterCreator.Create(Path.Combine(outputFolder, "DatabasesDescriptionQa.md"));
    }
    
    public void GenerateQaMdFile(List<DatabaseDescription> databasesDesc)
    {
        foreach (var database in databasesDesc)
        {
            Write_WhatAreTheTables(database);
            Write_WhatAreTheTablesUseJson(database);

            Write_WhatIsThePurposeOfDatabase(database);

            foreach (var table in database.Tables)
            {
                Write_WhatIsThePurposeOfTable(database, table);
                Write_ListAllTheColumnNamesInTheTable(database, table);
                Write_ListAllTheColumnDefinitionsInTheTable(database, table);
                
                foreach (var column in table.Columns)
                {
                    Write_WhatIsThePurposeOfColumn(database, table, column);
                }
            }
        }
    }

    private void Write_WhatIsThePurposeOfColumn(DatabaseDescription database, TableDescription table, ColumnDescription column)
    {
        if(column.Description.Trim().Length == 0)
        {
            return;
        }
        _writer.WriteLine(
            $"Question: What is the purpose or description of the {column.ColumnName} column in the {table.TableName} table of the {database.DatabaseName} database?");
        _writer.WriteLine($"Answer:");
        _writer.WriteLine($"{column.Description}");
        WriteDelimitLine();
    }

    private void Write_WhatIsThePurposeOfDatabase(DatabaseDescription database)
    {
        if (!string.IsNullOrEmpty(database.Description.Trim()))
        {
            _writer.WriteLine(
                $"Question: What is the purpose or description of the {database.DatabaseName} database?");
            _writer.WriteLine($"Answer:");
            _writer.WriteLine($"{database.Description}");
            WriteDelimitLine();
        }
    }

    private void Write_WhatAreTheTables(DatabaseDescription database)
    {
        _writer.WriteLine($"Question: What are the tables in the {database.DatabaseName} database?");
        _writer.WriteLine($"Answer:");
        foreach (var table in database.Tables)
        {
            _writer.WriteLine($"* {table.TableName}");
        }
        WriteDelimitLine();
    }

    private void Write_WhatAreTheTablesUseJson(DatabaseDescription database)
    {
        _writer.WriteLine(
            $"Question: What are the tables in the {database.DatabaseName} database? use JSON format response");
        _writer.WriteLine($"Answer:");
        _writer.WriteLine($"<|json|>");
        _writer.WriteLine($"```json");
        var tables = database.Tables.Select(x => new { x.TableName }).ToList();
        var json = JsonSerializer.Serialize(tables, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        _writer.WriteLine(json);
        _writer.WriteLine($"```");
        _writer.WriteLine($"<|end_json|>");
        WriteDelimitLine();
    }

    private void Write_WhatIsThePurposeOfTable(DatabaseDescription database, TableDescription table)
    {
        if (!string.IsNullOrEmpty(table.Description.Trim()))
        {
            _writer.WriteLine(
                $"Question: What is the purpose or description of the {table.TableName} table in the {database.DatabaseName} database?");
            _writer.WriteLine($"Answer:");
            _writer.WriteLine($"{table.Description}");
            WriteDelimitLine();
        }
    }

    private void Write_ListAllTheColumnNamesInTheTable(DatabaseDescription database, TableDescription table)
    {
        _writer.WriteLine($"Question: List all column names in the {table.TableName} table of the {database.DatabaseName} database.");
        _writer.WriteLine($"Answer:");
        foreach (var column in table.Columns)
        {
            _writer.WriteLine($"* {column.ColumnName}");
        }

        WriteDelimitLine();
    }

    private void Write_ListAllTheColumnDefinitionsInTheTable(DatabaseDescription database, TableDescription table)
    {
        _writer.WriteLine($"Question: List all the column definitions in the {table.TableName} table of the {database.DatabaseName} database.");
        _writer.WriteLine($"Answer:");
        foreach (var column in table.Columns)
        {
            _writer.Write($"* {column.ColumnName} {column.DataType}");
            if (column.IsNullable)
            {
                _writer.Write($",is Nullable");
            }
            if (column.IsIdentity)
            {
                _writer.Write($",is Identity");
            }

            if (!string.IsNullOrEmpty(column.DefaultValue))
            {
                _writer.Write($",Default Value: {column.DefaultValue}");
            }

            if (!string.IsNullOrEmpty(column.Description))
            {
                _writer.Write($",Description: {column.Description}");
            }
            _writer.WriteLine();
        }
        WriteDelimitLine();
    }

    private void WriteDelimitLine()
    {
        _writer.WriteLine();
        _writer.WriteLine();
        _writer.WriteLine();
    }

    public void Dispose()
    {
        _writer.Dispose();
    }
}