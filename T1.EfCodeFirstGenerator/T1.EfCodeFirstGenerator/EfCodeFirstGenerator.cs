using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using Newtonsoft.Json;
using T1.EfCodeFirstGenerator.CodeGenerator;
using T1.EfCodeFirstGenerator.Models;

namespace T1.EfCodeFirstGenerator
{
    [Generator]
    public class EfCodeFirstGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
            // No initialization required
        }

        public void Execute(GeneratorExecutionContext context)
        {
            var schemaFiles = context.AdditionalFiles
                .Where(f => f.Path.EndsWith(".schema", StringComparison.OrdinalIgnoreCase))
                .ToList();

            if (schemaFiles.Count == 0)
            {
                return;
            }

            foreach (var schemaFile in schemaFiles)
            {
                ProcessSchemaFile(context, schemaFile);
            }
        }

        private void ProcessSchemaFile(GeneratorExecutionContext context, AdditionalText schemaFile)
        {
            var dbSchema = LoadSchemaFromFile(schemaFile, context);

            if (dbSchema != null)
            {
                GenerateCodeFromSchema(context, dbSchema, schemaFile.Path);
            }
        }

        private DbSchema? LoadSchemaFromFile(AdditionalText schemaFile, GeneratorExecutionContext context)
        {
            try
            {
                var content = schemaFile.GetText(context.CancellationToken);
                if (content != null)
                {
                    var json = content.ToString();
                    return JsonConvert.DeserializeObject<DbSchema>(json);
                }
            }
            catch (Exception ex)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    new DiagnosticDescriptor(
                        "EFCFG002",
                        "Schema Load Error",
                        $"Failed to load schema from {schemaFile.Path}: {ex.Message}",
                        "EfCodeFirstGenerator",
                        DiagnosticSeverity.Warning,
                        true),
                    Location.None));
            }

            return null;
        }

        private void GenerateCodeFromSchema(GeneratorExecutionContext context, DbSchema dbSchema, string schemaFilePath)
        {
            try
            {
                var targetNamespace = DetermineNamespace(schemaFilePath);
                var generator = new EfCodeGenerator();
                var generatedFiles = generator.GenerateCodeFirstFromSchema(dbSchema, targetNamespace);

                foreach (var kvp in generatedFiles)
                {
                    var hintName = $"{dbSchema.DatabaseName}_{kvp.Key}";
                    context.AddSource(hintName, SourceText.From(kvp.Value, Encoding.UTF8));
                }
            }
            catch (Exception ex)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    new DiagnosticDescriptor(
                        "EFCFG003",
                        "Code Generation Error",
                        $"Failed to generate code for {dbSchema.DatabaseName}: {ex.Message}",
                        "EfCodeFirstGenerator",
                        DiagnosticSeverity.Warning,
                        true),
                    Location.None));
            }
        }

        private string DetermineNamespace(string schemaFilePath)
        {
            if (string.IsNullOrEmpty(schemaFilePath))
            {
                return "Generated";
            }

            var schemaDir = Path.GetDirectoryName(schemaFilePath) ?? "";
            var dirName = Path.GetFileName(schemaDir);
            
            if (string.IsNullOrEmpty(dirName))
            {
                return "Generated";
            }

            return $"Generated.{dirName}";
        }
    }
}

