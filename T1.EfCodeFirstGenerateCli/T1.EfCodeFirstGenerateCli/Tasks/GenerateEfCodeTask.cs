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

                DbConfigParser.ProcessAllConfigs(
                    ProjectDirectory,
                    ProcessDbConfig,
                    msg => Log.LogMessage(MessageImportance.Normal, msg)
                );

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
            var schemaFileName = $"{dbConfig.ContextName}.schema";
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
            
            // Add database-specific namespace layer to avoid naming conflicts
            var databaseNamespace = $"{targetNamespace}.Databases.{dbSchema.ContextName}";
            
            var generator = new EfCodeGenerator();
            var generatedFiles = generator.GenerateCodeFirstFromSchema(dbSchema, databaseNamespace);

            WriteGeneratedFiles(generatedFiles, generatedDir, dbSchema.DatabaseName);
            
            Log.LogMessage(MessageImportance.Normal, $"  Generated {generatedFiles.Count} file(s).");
        }

        private void WriteGeneratedFiles(System.Collections.Generic.Dictionary<string, string> generatedFiles, string generatedDir, string databaseName)
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
            
            CleanObsoleteFiles(generatedFiles, generatedDir, databaseName);
        }

        private void CleanObsoleteFiles(System.Collections.Generic.Dictionary<string, string> generatedFiles, string generatedDir, string databaseName)
        {
            var expectedFiles = new System.Collections.Generic.HashSet<string>(
                generatedFiles.Keys.Select(k => Path.GetFullPath(Path.Combine(generatedDir, k))),
                StringComparer.OrdinalIgnoreCase
            );
            
            // Only clean files in the current database directory
            var currentDbDir = Path.Combine(generatedDir, databaseName);
            if (!Directory.Exists(currentDbDir)) 
                return;
            
            var existingFiles = Directory.GetFiles(currentDbDir, "*", SearchOption.AllDirectories);
            foreach (var existingFile in existingFiles)
            {
                var fullPath = Path.GetFullPath(existingFile);
                
                // Skip .schema files
                if (Path.GetExtension(fullPath).Equals(".schema", StringComparison.OrdinalIgnoreCase))
                    continue;
                
                // Delete if not in expected files
                if (!expectedFiles.Contains(fullPath))
                {
                    File.Delete(fullPath);
                    Log.LogMessage(MessageImportance.Low, $"  Deleted obsolete file: {Path.GetRelativePath(generatedDir, fullPath)}");
                }
            }
            
            CleanEmptyDirectories(currentDbDir);
        }

        private void CleanEmptyDirectories(string directory)
        {
            foreach (var subDir in Directory.GetDirectories(directory))
            {
                CleanEmptyDirectories(subDir);
                if (!Directory.EnumerateFileSystemEntries(subDir).Any())
                {
                    Directory.Delete(subDir);
                }
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

