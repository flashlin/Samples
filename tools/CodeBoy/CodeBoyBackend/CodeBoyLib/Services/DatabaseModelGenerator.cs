using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using T1.Standard.IO;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Generated file information for EF Code First models
    /// </summary>
    public class EfGeneratedFile
    {
        /// <summary>
        /// Name of the generated file
        /// </summary>
        public string FileName { get; set; } = string.Empty;

        /// <summary>
        /// Content of the generated file
        /// </summary>
        public string FileContent { get; set; } = string.Empty;
    }

    /// <summary>
    /// Output structure for EF generation result
    /// </summary>
    public class EfGenerationOutput
    {
        /// <summary>
        /// Path to the generated .csproj file (filename with path)
        /// </summary>
        public string CsprojFilePath { get; set; } = string.Empty;

        /// <summary>
        /// List of generated code files
        /// </summary>
        public List<EfGeneratedFile> CodeFiles { get; set; } = new List<EfGeneratedFile>();
    }

    /// <summary>
    /// Parameters for EF Database First generation
    /// </summary>
    public class DatabaseGenerationParams
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
        /// Target framework version for the generated project
        /// </summary>
        public string TargetFrameworkVersion { get; set; } = "net9.0";

        /// <summary>
        /// Entity Framework version to use
        /// </summary>
        public string EFVersion { get; set; } = "9.0.8";
    }

    /// <summary>
    /// Result of EF Code First generation process
    /// </summary>
    public class EfGenerationResult
    {
        /// <summary>
        /// Whether the generation was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Generated files
        /// </summary>
        public List<EfGeneratedFile> GeneratedFiles { get; set; } = new List<EfGeneratedFile>();

        /// <summary>
        /// Path to the generated .csproj file
        /// </summary>
        public string CsprojFilePath { get; set; } = string.Empty;

        /// <summary>
        /// Any errors that occurred during generation
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Process log
        /// </summary>
        public List<string> ProcessLog { get; set; } = new List<string>();

        /// <summary>
        /// Temporary directory used for generation
        /// </summary>
        public string TempDirectory { get; set; } = string.Empty;

        /// <summary>
        /// Total duration of the process
        /// </summary>
        public TimeSpan Duration { get; set; }
    }

    /// <summary>
    /// Service for generating Entity Framework Code First models from existing databases
    /// </summary>
    public class DatabaseModelGenerator
    {
        /// <summary>
        /// Generates EF Code First models from database using scaffolding
        /// </summary>
        /// <param name="parameters">Database generation parameters</param>
        /// <returns>EF generation output containing csproj path and code files</returns>
        public async Task<EfGenerationOutput> GenerateEfCode(DatabaseGenerationParams parameters)
        {
            var result = await GenerateEfCodeWithResult(parameters);
            return new EfGenerationOutput
            {
                CsprojFilePath = result.CsprojFilePath,
                CodeFiles = result.GeneratedFiles
            };
        }

        /// <summary>
        /// Generates EF Code First models with detailed result information
        /// </summary>
        /// <param name="parameters">Database generation parameters</param>
        /// <returns>Detailed generation result</returns>
        public async Task<EfGenerationResult> GenerateEfCodeWithResult(DatabaseGenerationParams parameters)
        {
            var result = new EfGenerationResult();
            var startTime = DateTime.Now;

            try
            {
                result.ProcessLog.Add("🚀 Starting EF Code First generation from database...");
                
                // Validate parameters
                if (!ValidateParameters(parameters, result))
                {
                    return result;
                }

                // Create temporary directory for scaffolding
                result.TempDirectory = CreateTempDirectory();
                result.ProcessLog.Add($"📁 Created temp directory: {result.TempDirectory}");

                // Create minimal project structure for scaffolding
                await CreateScaffoldingProject(result.TempDirectory, parameters, result);

                // Execute EF scaffolding command
                var scaffoldSuccess = await ExecuteEfScaffolding(result.TempDirectory, parameters, result);
                
                if (scaffoldSuccess)
                {
                    // Read generated files
                    await ReadGeneratedFiles(result.TempDirectory, parameters, result);
                    result.Success = result.GeneratedFiles.Count > 0;
                }
                else
                {
                    result.Success = false;
                }

                result.Duration = DateTime.Now - startTime;
                result.ProcessLog.Add($"🏁 EF generation completed in {result.Duration.TotalSeconds:F2} seconds");

                return result;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Generation exception: {ex.Message}");
                result.ProcessLog.Add($"💥 Fatal error: {ex.Message}");
                result.Duration = DateTime.Now - startTime;
                return result;
            }
        }

        /// <summary>
        /// Validates input parameters
        /// </summary>
        private bool ValidateParameters(DatabaseGenerationParams parameters, EfGenerationResult result)
        {
            var errors = new List<string>();

            if (string.IsNullOrWhiteSpace(parameters.DatabaseServer))
                errors.Add("DatabaseServer is required");
            
            if (string.IsNullOrWhiteSpace(parameters.LoginId))
                errors.Add("LoginId is required");
            
            if (string.IsNullOrWhiteSpace(parameters.LoginPassword))
                errors.Add("LoginPassword is required");
            
            if (string.IsNullOrWhiteSpace(parameters.DatabaseName))
                errors.Add("DatabaseName is required");
            
            if (string.IsNullOrWhiteSpace(parameters.NamespaceName))
                errors.Add("NamespaceName is required");
            
            if (string.IsNullOrWhiteSpace(parameters.SdkName))
                errors.Add("SdkName is required");

            if (errors.Count > 0)
            {
                result.Errors.AddRange(errors);
                result.ProcessLog.Add("❌ Parameter validation failed");
                foreach (var error in errors)
                {
                    result.ProcessLog.Add($"   - {error}");
                }
                return false;
            }

            return true;
        }

        /// <summary>
        /// Creates a temporary directory for scaffolding operations
        /// </summary>
        private string CreateTempDirectory()
        {
            var tempBasePath = Path.GetTempPath();
            var randomName = $"EfScaffold_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
            var tempPath = Path.Combine(tempBasePath, randomName);
            
            Directory.CreateDirectory(tempPath);
            return Path.GetFullPath(tempPath);
        }

        /// <summary>
        /// Creates a minimal .NET project for scaffolding
        /// </summary>
        private async Task CreateScaffoldingProject(string tempDirectory, DatabaseGenerationParams parameters, EfGenerationResult result)
        {
            result.ProcessLog.Add("🔄 Creating scaffolding project...");

            // Create .csproj file with EF dependencies
            var csprojContent = GenerateScaffoldingCsproj(parameters);
            var csprojPath = Path.Combine(tempDirectory, $"{parameters.SdkName}.csproj");
            await File.WriteAllTextAsync(csprojPath, csprojContent);

            // Store the csproj file path in the result
            result.CsprojFilePath = csprojPath;

            result.ProcessLog.Add($"✅ Created project file: {csprojPath}");
        }

        /// <summary>
        /// Generates .csproj content for scaffolding
        /// </summary>
        private string GenerateScaffoldingCsproj(DatabaseGenerationParams parameters)
        {
            var output = new IndentStringBuilder();
            
            output.WriteLine("<Project Sdk=\"Microsoft.NET.Sdk\">");
            output.WriteLine();
            output.WriteLine("  <PropertyGroup>");
            output.WriteLine($"    <TargetFramework>{parameters.TargetFrameworkVersion}</TargetFramework>");
            output.WriteLine("    <ImplicitUsings>enable</ImplicitUsings>");
            output.WriteLine("    <Nullable>enable</Nullable>");
            output.WriteLine("  </PropertyGroup>");
            output.WriteLine();
            output.WriteLine("  <ItemGroup>");
            output.WriteLine($"    <PackageReference Include=\"Microsoft.EntityFrameworkCore\" Version=\"{parameters.EFVersion}\" />");
            output.WriteLine($"    <PackageReference Include=\"Microsoft.EntityFrameworkCore.SqlServer\" Version=\"{parameters.EFVersion}\" />");
            output.WriteLine($"    <PackageReference Include=\"Microsoft.EntityFrameworkCore.Tools\" Version=\"{parameters.EFVersion}\">");
            output.WriteLine("      <PrivateAssets>all</PrivateAssets>");
            output.WriteLine("      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>");
            output.WriteLine("    </PackageReference>");
            output.WriteLine($"    <PackageReference Include=\"Microsoft.EntityFrameworkCore.Design\" Version=\"{parameters.EFVersion}\">");
            output.WriteLine("      <PrivateAssets>all</PrivateAssets>");
            output.WriteLine("      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>");
            output.WriteLine("    </PackageReference>");
            output.WriteLine("  </ItemGroup>");
            output.WriteLine();
            output.WriteLine("</Project>");

            return output.ToString();
        }

        /// <summary>
        /// Executes EF scaffolding command
        /// </summary>
        private async Task<bool> ExecuteEfScaffolding(string tempDirectory, DatabaseGenerationParams parameters, EfGenerationResult result)
        {
            try
            {
                result.ProcessLog.Add("🔄 Executing EF scaffolding command...");

                // Build connection string
                var connectionString = BuildConnectionString(parameters);
                result.ProcessLog.Add("🔗 Built connection string (password hidden)");

                // Restore packages first
                result.ProcessLog.Add("📦 Restoring NuGet packages...");
                var restoreSuccess = await ExecuteDotnetCommand("restore", tempDirectory, result);
                if (!restoreSuccess)
                {
                    result.Errors.Add("Failed to restore NuGet packages");
                    return false;
                }

                // Prepare scaffolding command
                var scaffoldCommand = $"ef dbcontext scaffold \"{connectionString}\" Microsoft.EntityFrameworkCore.SqlServer " +
                                    $"--namespace \"{parameters.NamespaceName}\" " +
                                    $"--context-namespace \"{parameters.NamespaceName}\" " +
                                    $"--force " +
                                    $"--output-dir Models " +
                                    $"--context {parameters.SdkName}Context";

                result.ProcessLog.Add("⚡ Executing scaffolding command...");
                var scaffoldSuccess = await ExecuteDotnetCommand(scaffoldCommand, tempDirectory, result);

                if (scaffoldSuccess)
                {
                    result.ProcessLog.Add("✅ EF scaffolding completed successfully");
                    return true;
                }
                else
                {
                    result.Errors.Add("EF scaffolding command failed");
                    return false;
                }
            }
            catch (Exception ex)
            {
                result.Errors.Add($"Scaffolding execution error: {ex.Message}");
                result.ProcessLog.Add($"❌ Scaffolding failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Builds SQL Server connection string from parameters
        /// </summary>
        private string BuildConnectionString(DatabaseGenerationParams parameters)
        {
            return $"Server={parameters.DatabaseServer};Database={parameters.DatabaseName};" +
                   $"User Id={parameters.LoginId};Password={parameters.LoginPassword};" +
                   $"TrustServerCertificate=true;MultipleActiveResultSets=true";
        }

        /// <summary>
        /// Executes a dotnet command
        /// </summary>
        private async Task<bool> ExecuteDotnetCommand(string arguments, string workingDirectory, EfGenerationResult result)
        {
            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = new Process { StartInfo = startInfo };
                
                var outputLines = new List<string>();
                var errorLines = new List<string>();

                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        outputLines.Add(e.Data);
                    }
                };

                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        errorLines.Add(e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await process.WaitForExitAsync();

                // Log output
                foreach (var line in outputLines)
                {
                    result.ProcessLog.Add($"   {line}");
                }

                // Log errors if any
                if (errorLines.Count > 0)
                {
                    foreach (var line in errorLines)
                    {
                        result.ProcessLog.Add($"   ERROR: {line}");
                        result.Errors.Add(line);
                    }
                }

                return process.ExitCode == 0;
            }
            catch (Exception ex)
            {
                result.Errors.Add($"Command execution error: {ex.Message}");
                result.ProcessLog.Add($"❌ Command failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Reads generated files from the scaffolding output
        /// </summary>
        private async Task ReadGeneratedFiles(string tempDirectory, DatabaseGenerationParams parameters, EfGenerationResult result)
        {
            try
            {
                result.ProcessLog.Add("📖 Reading generated files...");

                var modelsDirectory = Path.Combine(tempDirectory, "Models");
                if (!Directory.Exists(modelsDirectory))
                {
                    result.ProcessLog.Add("❌ Models directory not found");
                    return;
                }

                // Read all .cs files
                var csFiles = Directory.GetFiles(modelsDirectory, "*.cs", SearchOption.AllDirectories);
                
                foreach (var filePath in csFiles)
                {
                    var fileName = Path.GetFileName(filePath);
                    var fileContent = await File.ReadAllTextAsync(filePath);
                    
                    // Apply namespace transformation if needed
                    fileContent = ApplyNamespaceTransformation(fileContent, parameters);
                    
                    result.GeneratedFiles.Add(new EfGeneratedFile
                    {
                        FileName = fileName,
                        FileContent = fileContent
                    });

                    result.ProcessLog.Add($"✅ Read file: {fileName} ({fileContent.Length} characters)");
                }

                // Also check for the context file in the root directory
                var contextFile = Path.Combine(tempDirectory, $"{parameters.SdkName}Context.cs");
                if (File.Exists(contextFile))
                {
                    var fileName = Path.GetFileName(contextFile);
                    var fileContent = await File.ReadAllTextAsync(contextFile);
                    fileContent = ApplyNamespaceTransformation(fileContent, parameters);
                    
                    result.GeneratedFiles.Add(new EfGeneratedFile
                    {
                        FileName = fileName,
                        FileContent = fileContent
                    });

                    result.ProcessLog.Add($"✅ Read context file: {fileName} ({fileContent.Length} characters)");
                }

                result.ProcessLog.Add($"📄 Total files read: {result.GeneratedFiles.Count}");
            }
            catch (Exception ex)
            {
                result.Errors.Add($"File reading error: {ex.Message}");
                result.ProcessLog.Add($"❌ Failed to read files: {ex.Message}");
            }
        }

        /// <summary>
        /// Applies namespace and other transformations to generated code
        /// </summary>
        private string ApplyNamespaceTransformation(string fileContent, DatabaseGenerationParams parameters)
        {
            // Replace temporary namespace with desired namespace
            fileContent = fileContent.Replace($"namespace {parameters.NamespaceName}", $"namespace {parameters.NamespaceName}");
            
            // Add any additional transformations here if needed
            // For example, adding version information, custom attributes, etc.
            
            return fileContent;
        }

        /// <summary>
        /// Cleans up the temporary directory
        /// </summary>
        public void CleanupTempDirectory(EfGenerationResult result)
        {
            try
            {
                if (!string.IsNullOrEmpty(result.TempDirectory) && Directory.Exists(result.TempDirectory))
                {
                    Directory.Delete(result.TempDirectory, true);
                    result.ProcessLog.Add($"🧹 Cleaned up temporary directory: {result.TempDirectory}");
                }
            }
            catch (Exception ex)
            {
                result.ProcessLog.Add($"⚠️  Warning: Could not clean up temp directory: {ex.Message}");
            }
        }

        /// <summary>
        /// Prints a detailed summary of the generation result
        /// </summary>
        public void PrintSummary(EfGenerationResult result)
        {
            Console.WriteLine("\n" + new string('=', 50));
            Console.WriteLine($"🗄️  EF Code First Generation Summary");
            Console.WriteLine(new string('=', 50));
            
            Console.WriteLine($"Success: {(result.Success ? "✅" : "❌")}");
            Console.WriteLine($"Duration: {result.Duration.TotalSeconds:F2} seconds");
            Console.WriteLine($"Generated Files: {result.GeneratedFiles.Count}");
            Console.WriteLine($"Temp Directory: {result.TempDirectory}");

            if (result.GeneratedFiles.Count > 0)
            {
                Console.WriteLine("\n📄 Generated Files:");
                foreach (var file in result.GeneratedFiles)
                {
                    Console.WriteLine($"   - {file.FileName} ({file.FileContent.Length} characters)");
                }
            }

            if (result.Errors.Count > 0)
            {
                Console.WriteLine("\n❌ Errors:");
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

            Console.WriteLine(new string('=', 50));
        }
    }
}
