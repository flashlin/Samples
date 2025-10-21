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

        public string? RootNamespace { get; set; }
        
        public string? AssemblyName { get; set; }

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

            // Get project base namespace
            var projectNamespace = GetProjectNamespace();
            
            // Calculate relative path from project root to .db file directory
            var relativeDbDir = Path.GetRelativePath(ProjectDirectory, dbFileDir);
            
            // Build target namespace
            string targetNamespace;
            if (string.IsNullOrEmpty(relativeDbDir) || relativeDbDir == ".")
            {
                // .db file is in project root
                targetNamespace = projectNamespace;
            }
            else
            {
                // .db file is in subdirectory
                var subNamespace = relativeDbDir.Replace(Path.DirectorySeparatorChar, '.')
                                                .Replace(Path.AltDirectorySeparatorChar, '.')
                                                .Trim('.');
                targetNamespace = $"{projectNamespace}.{subNamespace}";
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
                
                // Clear the Generated/{DatabaseName}/ directory before generating new code
                var databaseDir = Path.Combine(generatedDir, dbConfig.DatabaseName);
                if (Directory.Exists(databaseDir))
                {
                    try
                    {
                        Directory.Delete(databaseDir, true);
                        Log.LogMessage(MessageImportance.Low, $"  Cleared existing generated code directory.");
                    }
                    catch (Exception ex)
                    {
                        Log.LogWarning($"  Warning: Failed to clear directory {databaseDir}: {ex.Message}");
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

                Log.LogMessage(MessageImportance.Normal, $"  Generated {generatedFiles.Count} file(s).");
            }
            catch (Exception ex)
            {
                Log.LogWarning($"  Failed to generate code for {dbConfig.DatabaseName}: {ex.Message}");
            }
        }

        public string SanitizeFileName(string fileName)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return string.Join("_", fileName.Split(invalid, StringSplitOptions.RemoveEmptyEntries));
        }

        public string GetProjectNamespace()
        {
            // Priority: RootNamespace → AssemblyName → Project directory name
            if (!string.IsNullOrEmpty(RootNamespace))
                return RootNamespace;
            
            if (!string.IsNullOrEmpty(AssemblyName))
                return AssemblyName;
            
            // Fallback to project directory name
            var projectDirName = new DirectoryInfo(ProjectDirectory).Name;
            return string.IsNullOrEmpty(projectDirName) ? "Generated" : projectDirName;
        }
    }
}

