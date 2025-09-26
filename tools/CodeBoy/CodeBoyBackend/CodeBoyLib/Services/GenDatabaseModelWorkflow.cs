using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Parameters for database model generation workflow
    /// </summary>
    public class GenDatabaseModelBuildParams
    {
        /// <summary>
        /// Database server connection string or server name
        /// </summary>
        public required string DatabaseServer { get; set; }

        /// <summary>
        /// Login ID for database authentication
        /// </summary>
        public required string LoginId { get; set; }

        /// <summary>
        /// Login password for database authentication
        /// </summary>
        public required string LoginPassword { get; set; }

        /// <summary>
        /// Database name to scaffold
        /// </summary>
        public required string DatabaseName { get; set; }

        /// <summary>
        /// Namespace name for generated models
        /// </summary>
        public required string NamespaceName { get; set; }

        /// <summary>
        /// SDK name for the generated package
        /// </summary>
        public required string SdkName { get; set; }

        /// <summary>
        /// SDK version for the generated package
        /// </summary>
        public string SdkVersion { get; set; } = "1.0.0";

        /// <summary>
        /// List of target frameworks to generate for
        /// </summary>
        public List<string> TargetFrameworks { get; set; } = new List<string> { "net8.0", "net9.0" };

        /// <summary>
        /// Entity Framework version to use
        /// </summary>
        public string EFVersion { get; set; } = "9.0.8";

        /// <summary>
        /// Base output path for all build operations
        /// </summary>
        public required string OutputPath { get; set; }
    }

    /// <summary>
    /// Result of the complete database model generation and build process
    /// </summary>
    public class GenDatabaseModelResult
    {
        /// <summary>
        /// Whether the entire process was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// SDK name that was generated
        /// </summary>
        public string SdkName { get; set; } = string.Empty;

        /// <summary>
        /// Path to the generated NuGet package file
        /// </summary>
        public string NupkgFile { get; set; } = string.Empty;

        /// <summary>
        /// List of successful project paths for each target framework
        /// </summary>
        public List<string> ProjectPaths { get; set; } = new List<string>();

        /// <summary>
        /// Any errors that occurred during the process
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Total duration of the entire process
        /// </summary>
        public TimeSpan TotalDuration { get; set; }

        /// <summary>
        /// Step-by-step process log
        /// </summary>
        public List<string> ProcessLog { get; set; } = new List<string>();

        /// <summary>
        /// Target frameworks that were successfully built
        /// </summary>
        public List<string> SuccessfulFrameworks { get; set; } = new List<string>();

        /// <summary>
        /// Target frameworks that failed to build
        /// </summary>
        public List<string> FailedFrameworks { get; set; } = new List<string>();
    }

    /// <summary>
    /// Workflow service for generating complete database model projects with multiple target frameworks
    /// </summary>
    public class GenDatabaseModelWorkflow : IGenDatabaseModelWorkflow
    {
        private readonly IDatabaseModelGenerator _databaseModelGenerator;
        private readonly NupkgFileGenerator _nupkgGenerator;
        private readonly ILogger<GenDatabaseModelWorkflow> _logger;

        /// <summary>
        /// Initializes a new instance of GenDatabaseModelWorkflow
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="databaseModelGenerator">Database model generator service</param>
        /// <param name="nupkgGenerator">NuGet package generator service</param>
        public GenDatabaseModelWorkflow(
            ILogger<GenDatabaseModelWorkflow> logger, 
            IDatabaseModelGenerator databaseModelGenerator,
            NupkgFileGenerator nupkgGenerator)
        {
            _logger = logger;
            _databaseModelGenerator = databaseModelGenerator;
            _nupkgGenerator = nupkgGenerator;
        }

        /// <summary>
        /// Builds database models for multiple target frameworks and generates a NuGet package
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <returns>Generation and build result</returns>
        public async Task<GenDatabaseModelResult> Build(GenDatabaseModelBuildParams buildParams)
        {
            var result = new GenDatabaseModelResult
            {
                SdkName = buildParams.SdkName
            };

            var startTime = DateTime.Now;

            try
            {
                result.ProcessLog.Add($"🚀 Starting multi-target database model generation for frameworks: {string.Join(", ", buildParams.TargetFrameworks)}");
                result.ProcessLog.Add($"📁 Using output path: {buildParams.OutputPath}");
                result.ProcessLog.Add($"🗄️  Database: {buildParams.DatabaseName} on {buildParams.DatabaseServer}");

                // Ensure output directory exists
                Directory.CreateDirectory(buildParams.OutputPath);

                // Build for each target framework
                await BuildAllFrameworks(buildParams, result);

                // Generate NuGet package if any frameworks were successful
                if (result.ProjectPaths.Count > 0)
                {
                    await GenerateNuGetPackage(buildParams, result);
                }
                else
                {
                    result.ProcessLog.Add("❌ No successful frameworks to package");
                    result.Errors.Add("No target frameworks were successfully built");
                }

                // Finalize the result
                FinalizeResult(result, buildParams.TargetFrameworks, startTime);

                return result;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Workflow exception: {ex.Message}");
                result.ProcessLog.Add($"💥 Fatal error: {ex.Message}");
                result.TotalDuration = DateTime.Now - startTime;
                return result;
            }
        }

        /// <summary>
        /// Builds database models for all target frameworks sequentially
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <param name="result">Main result object to update</param>
        private async Task BuildAllFrameworks(GenDatabaseModelBuildParams buildParams, GenDatabaseModelResult result)
        {
            foreach (var framework in buildParams.TargetFrameworks)
            {
                result.ProcessLog.Add($"🔄 Building database models for {framework}...");
                
                var frameworkSuccess = await BuildSingleFramework(buildParams, framework, result);
                
                if (frameworkSuccess)
                {
                    result.SuccessfulFrameworks.Add(framework);
                    result.ProcessLog.Add($"✅ {framework} build successful");
                }
                else
                {
                    result.FailedFrameworks.Add(framework);
                    result.ProcessLog.Add($"❌ {framework} build failed");
                }
            }
        }

        /// <summary>
        /// Builds database models for a single target framework
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <param name="framework">Target framework (e.g., "net8.0")</param>
        /// <param name="result">Main result object to update</param>
        /// <returns>True if successful, false otherwise</returns>
        private async Task<bool> BuildSingleFramework(GenDatabaseModelBuildParams buildParams, string framework, GenDatabaseModelResult result)
        {
            try
            {
                // Create framework-specific parameters
                var generationParams = CreateFrameworkGenerationParams(buildParams, framework);

                // Generate EF models using DatabaseModelGenerator
                var generationOutput = await _databaseModelGenerator.GenerateEfCode(generationParams);

                if (generationOutput.CodeFiles.Count > 0)
                {
                    // Get the directory containing the csproj file
                    var projectDirectory = Path.GetDirectoryName(generationOutput.CsprojFilePath);
                    if (!string.IsNullOrEmpty(projectDirectory))
                    {
                        result.ProjectPaths.Add(projectDirectory);
                        result.ProcessLog.Add($"📁 Added project path: {projectDirectory}");
                        return true;
                    }
                    else
                    {
                        result.ProcessLog.Add($"❌ Could not determine project directory from: {generationOutput.CsprojFilePath}");
                        return false;
                    }
                }
                else
                {
                    result.ProcessLog.Add($"❌ No code files generated for {framework}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                result.ProcessLog.Add($"❌ {framework} build failed with exception: {ex.Message}");
                result.Errors.Add($"{framework} build exception: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Creates framework-specific generation parameters
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <param name="framework">Target framework</param>
        /// <returns>DatabaseGenerationParams for the framework</returns>
        private DatabaseGenerationParams CreateFrameworkGenerationParams(GenDatabaseModelBuildParams buildParams, string framework)
        {
            return new DatabaseGenerationParams
            {
                DatabaseServer = buildParams.DatabaseServer,
                LoginId = buildParams.LoginId,
                LoginPassword = buildParams.LoginPassword,
                DatabaseName = buildParams.DatabaseName,
                NamespaceName = buildParams.NamespaceName,
                SdkName = buildParams.SdkName,
                SdkVersion = buildParams.SdkVersion,
                TargetFrameworkVersion = framework,
                EFVersion = buildParams.EFVersion
            };
        }

        /// <summary>
        /// Generates NuGet package from successful project builds
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <param name="result">Main result object to update</param>
        private async Task GenerateNuGetPackage(GenDatabaseModelBuildParams buildParams, GenDatabaseModelResult result)
        {
            try
            {
                result.ProcessLog.Add("🔄 Generating NuGet package...");
                
                var nupkgName = $"Titansoft.{buildParams.SdkName}";
                var nupkgFile = Path.Combine(buildParams.OutputPath, $"{nupkgName}.{buildParams.SdkVersion}.nupkg");
                
                result.ProcessLog.Add($"📦 Package name: {nupkgName}");
                result.ProcessLog.Add($"📦 Package file: {nupkgFile}");
                result.ProcessLog.Add($"📁 Including {result.ProjectPaths.Count} project(s)");

                var nupkgSuccess = _nupkgGenerator.Generate(nupkgFile, result.ProjectPaths, buildParams.SdkVersion);
                
                if (nupkgSuccess)
                {
                    result.NupkgFile = nupkgFile;
                    result.ProcessLog.Add($"✅ NuGet package created successfully: {nupkgFile}");
                }
                else
                {
                    result.ProcessLog.Add("❌ Failed to create NuGet package");
                    result.Errors.Add("NuGet package generation failed");
                }
            }
            catch (Exception ex)
            {
                result.ProcessLog.Add($"❌ NuGet package generation failed: {ex.Message}");
                result.Errors.Add($"NuGet package generation exception: {ex.Message}");
            }

            await Task.CompletedTask; // Make method async for consistency
        }

        /// <summary>
        /// Finalizes the result and determines overall success
        /// </summary>
        /// <param name="result">Result object to finalize</param>
        /// <param name="targetFrameworks">Array of target frameworks that were attempted</param>
        /// <param name="startTime">Start time of the build process</param>
        private void FinalizeResult(GenDatabaseModelResult result, List<string> targetFrameworks, DateTime startTime)
        {
            // Determine overall success
            result.Success = result.SuccessfulFrameworks.Count > 0 && !string.IsNullOrEmpty(result.NupkgFile);
            result.TotalDuration = DateTime.Now - startTime;
            
            result.ProcessLog.Add($"🏁 Multi-target database model generation completed in {result.TotalDuration.TotalSeconds:F2} seconds");
            result.ProcessLog.Add($"📊 Successful frameworks: {result.SuccessfulFrameworks.Count}/{targetFrameworks.Count}");
            
            if (result.SuccessfulFrameworks.Count > 0)
            {
                result.ProcessLog.Add($"✅ Successful: {string.Join(", ", result.SuccessfulFrameworks)}");
            }
            
            if (result.FailedFrameworks.Count > 0)
            {
                result.ProcessLog.Add($"❌ Failed: {string.Join(", ", result.FailedFrameworks)}");
            }

            if (result.Success)
            {
                result.ProcessLog.Add($"🎉 Overall status: SUCCESS - NuGet package ready at {result.NupkgFile}");
            }
            else
            {
                result.ProcessLog.Add("💥 Overall status: FAILED");
            }
        }

        /// <summary>
        /// Prints a detailed summary of the generation result
        /// </summary>
        /// <param name="result">Generation result to summarize</param>
        public void PrintSummary(GenDatabaseModelResult result)
        {
            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine($"🗄️  Database Model Generation Summary");
            Console.WriteLine(new string('=', 60));
            
            Console.WriteLine($"SDK Name: {result.SdkName}");
            Console.WriteLine($"Success: {(result.Success ? "✅" : "❌")}");
            Console.WriteLine($"Duration: {result.TotalDuration.TotalSeconds:F2} seconds");
            Console.WriteLine($"NuGet Package: {result.NupkgFile}");
            Console.WriteLine($"Project Paths: {result.ProjectPaths.Count}");

            if (result.SuccessfulFrameworks.Count > 0)
            {
                Console.WriteLine($"\n✅ Successful Frameworks ({result.SuccessfulFrameworks.Count}):");
                foreach (var framework in result.SuccessfulFrameworks)
                {
                    Console.WriteLine($"   - {framework}");
                }
            }

            if (result.FailedFrameworks.Count > 0)
            {
                Console.WriteLine($"\n❌ Failed Frameworks ({result.FailedFrameworks.Count}):");
                foreach (var framework in result.FailedFrameworks)
                {
                    Console.WriteLine($"   - {framework}");
                }
            }

            if (result.ProjectPaths.Count > 0)
            {
                Console.WriteLine($"\n📁 Generated Project Paths:");
                foreach (var path in result.ProjectPaths)
                {
                    Console.WriteLine($"   - {path}");
                }
            }

            if (result.Errors.Count > 0)
            {
                Console.WriteLine($"\n❌ Errors ({result.Errors.Count}):");
                foreach (var error in result.Errors)
                {
                    Console.WriteLine($"   - {error}");
                }
            }

            Console.WriteLine("\n📝 Process Log:");
            foreach (var logEntry in result.ProcessLog)
            {
                Console.WriteLine($"   {logEntry}");
            }

            Console.WriteLine(new string('=', 60));
        }

        /// <summary>
        /// Cleans up temporary directories for a result
        /// </summary>
        /// <param name="result">Result containing project paths to clean</param>
        public void CleanupTempDirectories(GenDatabaseModelResult result)
        {
            foreach (var projectPath in result.ProjectPaths)
            {
                try
                {
                    if (Directory.Exists(projectPath))
                    {
                        Directory.Delete(projectPath, true);
                        Console.WriteLine($"🧹 Cleaned up temporary directory: {projectPath}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️  Warning: Could not clean up temp directory {projectPath}: {ex.Message}");
                }
            }
        }
    }
}
