using System;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using T1.EfCodeFirstGenerator.CLI.ConfigParser;
using T1.EfCodeFirstGenerator.CLI.SchemaExtractor;

namespace T1.EfCodeFirstGenerator.CLI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("T1.EfCodeFirstGenerator CLI - Schema Extractor");
            Console.WriteLine("==============================================");
            Console.WriteLine();

            string targetDirectory = args.Length > 0 ? args[0] : Directory.GetCurrentDirectory();

            if (!Directory.Exists(targetDirectory))
            {
                Console.WriteLine($"Error: Directory '{targetDirectory}' does not exist.");
                return;
            }

            Console.WriteLine($"Scanning directory: {targetDirectory}");
            Console.WriteLine();

            var dbConfigs = DbConfigParser.GetAllDbConnectionConfigs(targetDirectory);

            if (dbConfigs.Count == 0)
            {
                Console.WriteLine("No .db files found or no valid connection strings.");
                return;
            }

            Console.WriteLine($"Found {dbConfigs.Count} database configuration(s).");
            Console.WriteLine();

            foreach (var dbConfig in dbConfigs)
            {
                ProcessDbConfig(dbConfig, targetDirectory);
            }

            Console.WriteLine();
            Console.WriteLine("Schema extraction completed.");
        }

        static void ProcessDbConfig(Models.DbConfig dbConfig, string targetDirectory)
        {
            Console.WriteLine($"Processing: {dbConfig.ServerName}/{dbConfig.DatabaseName}");

            var schemaFileName = $"{SanitizeFileName(dbConfig.ServerName)}_{SanitizeFileName(dbConfig.DatabaseName)}.schema";
            var schemaFilePath = Path.Combine(targetDirectory, schemaFileName);

            if (File.Exists(schemaFilePath))
            {
                Console.WriteLine($"  Schema file already exists: {schemaFileName}");
                Console.WriteLine($"  Skipping... (delete the file to regenerate)");
                Console.WriteLine();
                return;
            }

            try
            {
                Console.WriteLine($"  Connecting to database...");
                var dbSchema = DatabaseSchemaExtractor.CreateDatabaseSchema(dbConfig);

                Console.WriteLine($"  Extracted {dbSchema.Tables.Count} table(s).");

                var json = JsonConvert.SerializeObject(dbSchema, Formatting.Indented);
                File.WriteAllText(schemaFilePath, json, Encoding.UTF8);

                Console.WriteLine($"  Schema saved to: {schemaFileName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error: {ex.Message}");
            }

            Console.WriteLine();
        }

        static string SanitizeFileName(string fileName)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return string.Join("_", fileName.Split(invalid, StringSplitOptions.RemoveEmptyEntries));
        }
    }
}

