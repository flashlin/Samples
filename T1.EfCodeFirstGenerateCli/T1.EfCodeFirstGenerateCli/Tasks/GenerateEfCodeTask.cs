using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using System;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using T1.EfCodeFirstGenerateCli.ConfigParser;
using T1.EfCodeFirstGenerateCli.SchemaExtractor;
using T1.EfCodeFirstGenerateCli.CodeGenerator;

namespace T1.EfCodeFirstGenerateCli.Tasks
{
    public class GenerateEfCodeTask : Task
    {
        [Required]
        public string ProjectDirectory { get; set; } = string.Empty;

        public override bool Execute()
        {
            try
            {
                Log.LogMessage(MessageImportance.High, "T1.EfCodeFirstGenerateCli: Starting code generation...");

                if (!Directory.Exists(ProjectDirectory))
                {
                    Log.LogWarning($"Project directory not found: {ProjectDirectory}");
                    return true; // Not a failure, just skip
                }

                var dbConfigs = DbConfigParser.GetAllDbConnectionConfigs(ProjectDirectory);

                if (dbConfigs.Count == 0)
                {
                    Log.LogMessage(MessageImportance.Low, "No .db files found or no valid connection strings.");
                    return true;
                }

                Log.LogMessage(MessageImportance.Normal, $"Found {dbConfigs.Count} database configuration(s).");

                foreach (var dbConfig in dbConfigs)
                {
                    ProcessDbConfig(dbConfig);
                }

                Log.LogMessage(MessageImportance.High, "T1.EfCodeFirstGenerateCli: Code generation completed.");
                return true;
            }
            catch (Exception ex)
            {
                Log.LogError($"Error during code generation: {ex.Message}");
                Log.LogErrorFromException(ex, true);
                return false;
            }
        }

        private void ProcessDbConfig(Models.DbConfig dbConfig)
        {
            Log.LogMessage(MessageImportance.Normal, $"Processing: {dbConfig.ServerName}/{dbConfig.DatabaseName}");

            // Get the directory where the .db file is located
            var dbFileDir = Path.GetDirectoryName(dbConfig.DbFilePath);
            if (string.IsNullOrEmpty(dbFileDir))
            {
                Log.LogWarning($"  Unable to determine .db file directory.");
                return;
            }

            // Calculate namespace based on the relative path from project root to .db file directory
            var relativeDbDir = Path.GetRelativePath(ProjectDirectory, dbFileDir);
            var targetNamespace = relativeDbDir.Replace(Path.DirectorySeparatorChar, '.').Replace(Path.AltDirectorySeparatorChar, '.');
            
            // Remove leading/trailing dots
            targetNamespace = targetNamespace.Trim('.');
            
            // If the .db file is in the project root, use the project directory name as namespace
            if (string.IsNullOrEmpty(targetNamespace) || targetNamespace == ".")
            {
                var projectDirName = new DirectoryInfo(ProjectDirectory).Name;
                targetNamespace = string.IsNullOrEmpty(projectDirName) ? "Generated" : projectDirName;
            }

            Log.LogMessage(MessageImportance.Normal, $"  Target namespace: {targetNamespace}");

            var schemaFileName = $"{SanitizeFileName(dbConfig.ServerName)}_{SanitizeFileName(dbConfig.DatabaseName)}.schema";
            var generatedDir = Path.Combine(dbFileDir, "Generated");
            Directory.CreateDirectory(generatedDir);

            var schemaFilePath = Path.Combine(generatedDir, schemaFileName);

            Models.DbSchema dbSchema;

            if (File.Exists(schemaFilePath))
            {
                Log.LogMessage(MessageImportance.Low, $"  Loading existing schema: {schemaFileName}");
                var json = File.ReadAllText(schemaFilePath, Encoding.UTF8);
                dbSchema = JsonConvert.DeserializeObject<Models.DbSchema>(json)!;
            }
            else
            {
                try
                {
                    Log.LogMessage(MessageImportance.Normal, $"  Connecting to database...");
                    dbSchema = DatabaseSchemaExtractor.CreateDatabaseSchema(dbConfig);

                    Log.LogMessage(MessageImportance.Normal, $"  Extracted {dbSchema.Tables.Count} table(s).");

                    var json = JsonConvert.SerializeObject(dbSchema, Formatting.Indented);
                    File.WriteAllText(schemaFilePath, json, Encoding.UTF8);

                    Log.LogMessage(MessageImportance.Normal, $"  Schema saved to: {schemaFileName}");
                }
                catch (Exception ex)
                {
                    Log.LogWarning($"  Failed to extract schema for {dbConfig.DatabaseName}: {ex.Message}");
                    return;
                }
            }

            // Generate EF Core code
            try
            {
                Log.LogMessage(MessageImportance.Low, $"  Generating EF Core code...");
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

                Log.LogMessage(MessageImportance.Normal, $"  Generated {generatedFiles.Count} file(s).");
            }
            catch (Exception ex)
            {
                Log.LogWarning($"  Failed to generate code for {dbConfig.DatabaseName}: {ex.Message}");
            }
        }

        private string SanitizeFileName(string fileName)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return string.Join("_", fileName.Split(invalid, StringSplitOptions.RemoveEmptyEntries));
        }
    }
}

