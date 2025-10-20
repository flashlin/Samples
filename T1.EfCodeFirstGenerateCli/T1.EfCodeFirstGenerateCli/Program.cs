using System;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using T1.EfCodeFirstGenerateCli.ConfigParser;
using T1.EfCodeFirstGenerateCli.SchemaExtractor;
using T1.EfCodeFirstGenerateCli.CodeGenerator;

namespace T1.EfCodeFirstGenerateCli
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("T1.EfCodeFirstGenerateCli - EF Core Code Generator");
            Console.WriteLine("================================================");
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
            Console.WriteLine("Code generation completed.");
        }

        static void ProcessDbConfig(Models.DbConfig dbConfig, string projectRoot)
        {
            Console.WriteLine($"Processing: {dbConfig.ServerName}/{dbConfig.DatabaseName}");

            // Get the directory where the .db file is located
            var dbFileDir = Path.GetDirectoryName(dbConfig.DbFilePath);
            if (string.IsNullOrEmpty(dbFileDir))
            {
                Console.WriteLine($"  Error: Unable to determine .db file directory.");
                Console.WriteLine();
                return;
            }

            // Calculate namespace based on the relative path from project root to .db file directory
            var relativeDbDir = Path.GetRelativePath(projectRoot, dbFileDir);
            var targetNamespace = relativeDbDir.Replace(Path.DirectorySeparatorChar, '.').Replace(Path.AltDirectorySeparatorChar, '.');
            
            // Remove leading/trailing dots
            targetNamespace = targetNamespace.Trim('.');
            
            // If the .db file is in the project root, use the project directory name as namespace
            if (string.IsNullOrEmpty(targetNamespace) || targetNamespace == ".")
            {
                var projectDirName = new DirectoryInfo(projectRoot).Name;
                targetNamespace = string.IsNullOrEmpty(projectDirName) ? "Generated" : projectDirName;
            }

            Console.WriteLine($"  Target namespace: {targetNamespace}");

            var schemaFileName = $"{SanitizeFileName(dbConfig.ServerName)}_{SanitizeFileName(dbConfig.DatabaseName)}.schema";
            var generatedDir = Path.Combine(dbFileDir, "Generated");
            Directory.CreateDirectory(generatedDir);
            
            var schemaFilePath = Path.Combine(generatedDir, schemaFileName);

            Models.DbSchema dbSchema;

            if (File.Exists(schemaFilePath))
            {
                Console.WriteLine($"  Loading existing schema: {schemaFileName}");
                var json = File.ReadAllText(schemaFilePath, Encoding.UTF8);
                dbSchema = JsonConvert.DeserializeObject<Models.DbSchema>(json)!;
            }
            else
            {
                try
                {
                    Console.WriteLine($"  Connecting to database...");
                    dbSchema = DatabaseSchemaExtractor.CreateDatabaseSchema(dbConfig);

                    Console.WriteLine($"  Extracted {dbSchema.Tables.Count} table(s).");

                    var json = JsonConvert.SerializeObject(dbSchema, Formatting.Indented);
                    File.WriteAllText(schemaFilePath, json, Encoding.UTF8);

                    Console.WriteLine($"  Schema saved to: {schemaFileName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error: {ex.Message}");
                    Console.WriteLine();
                    return;
                }
            }

            // Generate EF Core code
            try
            {
                Console.WriteLine($"  Generating EF Core code...");
                
                // Clear the Generated/{DatabaseName}/ directory before generating new code
                var databaseDir = Path.Combine(generatedDir, dbConfig.DatabaseName);
                if (Directory.Exists(databaseDir))
                {
                    try
                    {
                        Directory.Delete(databaseDir, true);
                        Console.WriteLine($"  Cleared existing generated code directory.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Warning: Failed to clear directory {databaseDir}: {ex.Message}");
                    }
                }
                
                var generator = new EfCodeGenerator();
                var generatedFiles = generator.GenerateCodeFirstFromSchema(dbSchema, targetNamespace);

                foreach (var kvp in generatedFiles)
                {
                    var filePath = Path.Combine(generatedDir, kvp.Key);
                    var fileDir = Path.GetDirectoryName(filePath);
                    if (!string.IsNullOrEmpty(fileDir))
                    {
                        Directory.CreateDirectory(fileDir);
                    }
                    File.WriteAllText(filePath, kvp.Value, Encoding.UTF8);
                }

                Console.WriteLine($"  Generated {generatedFiles.Count} file(s).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error generating code: {ex.Message}");
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

