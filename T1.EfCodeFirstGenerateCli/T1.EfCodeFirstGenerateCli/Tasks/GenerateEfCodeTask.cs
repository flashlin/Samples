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

            var dbFileDir = GetDbFileDirectory(dbConfig);
            if (string.IsNullOrEmpty(dbFileDir))
                return;

            var targetNamespace = CalculateTargetNamespace(dbFileDir);
            Log.LogMessage(MessageImportance.Normal, $"  Target namespace: {targetNamespace}");

            var generatedDir = Path.Combine(dbFileDir, "Generated");
            Directory.CreateDirectory(generatedDir);

            var dbSchema = LoadOrExtractSchema(dbConfig, generatedDir);
            if (dbSchema == null)
                return;

            GenerateEfCoreCode(dbConfig, dbSchema, targetNamespace, generatedDir);
        }

        private string? GetDbFileDirectory(Models.DbConfig dbConfig)
        {
            var dbFileDir = Path.GetDirectoryName(dbConfig.DbFilePath);
            if (string.IsNullOrEmpty(dbFileDir))
            {
                Log.LogWarning($"  Unable to determine .db file directory.");
                return null;
            }
            return dbFileDir;
        }

        private string CalculateTargetNamespace(string dbFileDir)
        {
            var projectNamespace = GetProjectNamespace();
            var relativeDbDir = Path.GetRelativePath(ProjectDirectory, dbFileDir);
            
            if (string.IsNullOrEmpty(relativeDbDir) || relativeDbDir == ".")
            {
                return projectNamespace;
            }
            
            var subNamespace = relativeDbDir.Replace(Path.DirectorySeparatorChar, '.')
                                            .Replace(Path.AltDirectorySeparatorChar, '.')
                                            .Trim('.');
            return $"{projectNamespace}.{subNamespace}";
        }

        private Models.DbSchema? LoadOrExtractSchema(Models.DbConfig dbConfig, string generatedDir)
        {
            var schemaFileName = $"{SanitizeFileName(dbConfig.ServerName)}_{SanitizeFileName(dbConfig.DatabaseName)}.schema";
            var schemaFilePath = Path.Combine(generatedDir, schemaFileName);

            if (File.Exists(schemaFilePath))
            {
                return LoadExistingSchema(schemaFilePath, schemaFileName);
            }
            
            return ExtractAndSaveSchema(dbConfig, schemaFilePath, schemaFileName);
        }

        private Models.DbSchema LoadExistingSchema(string schemaFilePath, string schemaFileName)
        {
            Log.LogMessage(MessageImportance.Low, $"  Loading existing schema: {schemaFileName}");
            var json = File.ReadAllText(schemaFilePath, Encoding.UTF8);
            return JsonConvert.DeserializeObject<Models.DbSchema>(json)!;
        }

        private Models.DbSchema? ExtractAndSaveSchema(Models.DbConfig dbConfig, string schemaFilePath, string schemaFileName)
        {
            Log.LogMessage(MessageImportance.Normal, $"  Connecting to database...");
            var dbSchema = DatabaseSchemaExtractor.CreateDatabaseSchema(dbConfig);

            Log.LogMessage(MessageImportance.Normal, $"  Extracted {dbSchema.Tables.Count} table(s).");

            var json = JsonConvert.SerializeObject(dbSchema, Formatting.Indented);
            File.WriteAllText(schemaFilePath, json, Encoding.UTF8);

            Log.LogMessage(MessageImportance.Normal, $"  Schema saved to: {schemaFileName}");
            return dbSchema;
        }

        private void GenerateEfCoreCode(Models.DbConfig dbConfig, Models.DbSchema dbSchema, string targetNamespace, string generatedDir)
        {
            Log.LogMessage(MessageImportance.Low, $"  Generating EF Core code...");
            
            ClearGeneratedDirectory(dbConfig, generatedDir);
            
            var generator = new EfCodeGenerator();
            var generatedFiles = generator.GenerateCodeFirstFromSchema(dbSchema, targetNamespace);

            WriteGeneratedFiles(generatedFiles, generatedDir);
            
            Log.LogMessage(MessageImportance.Normal, $"  Generated {generatedFiles.Count} file(s).");
        }

        private void ClearGeneratedDirectory(Models.DbConfig dbConfig, string generatedDir)
        {
            var databaseDir = Path.Combine(generatedDir, dbConfig.DatabaseName);
            if (Directory.Exists(databaseDir))
            {
                Directory.Delete(databaseDir, true);
                Log.LogMessage(MessageImportance.Low, $"  Cleared existing generated code directory.");
            }
        }

        private void WriteGeneratedFiles(System.Collections.Generic.Dictionary<string, string> generatedFiles, string generatedDir)
        {
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

