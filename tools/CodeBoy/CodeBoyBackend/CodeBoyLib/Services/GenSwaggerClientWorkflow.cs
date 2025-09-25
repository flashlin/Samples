using CodeBoyLib.Models;
using System;
using System.IO;
using System.Threading.Tasks;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Result of the complete Swagger client generation and build process
    /// </summary>
    public class GenSwaggerClientResult
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
        /// Temporary directory where files were generated
        /// </summary>
        public string TempDirectory { get; set; } = string.Empty;

        /// <summary>
        /// Path to the generated C# client code file
        /// </summary>
        public string ClientCodePath { get; set; } = string.Empty;

        /// <summary>
        /// Path to the generated .csproj file
        /// </summary>
        public string CsprojPath { get; set; } = string.Empty;

        /// <summary>
        /// Path to the built assembly (if build was successful)
        /// </summary>
        public string? AssemblyPath { get; set; }

        /// <summary>
        /// Code generation result
        /// </summary>
        public string GeneratedCode { get; set; } = string.Empty;

        /// <summary>
        /// Build result details
        /// </summary>
        public BuildResult? BuildResult { get; set; }

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
    }

    /// <summary>
    /// Configuration for the Swagger client generation process
    /// </summary>
    public class GenSwaggerClientConfig
    {
        /// <summary>
        /// Target .NET version for the generated project
        /// </summary>
        public string DotnetVersion { get; set; } = "net8.0";

        /// <summary>
        /// SDK version for the generated package
        /// </summary>
        public string SdkVersion { get; set; } = "1.0.0";

        /// <summary>
        /// Build configuration (Debug/Release)
        /// </summary>
        public string BuildConfiguration { get; set; } = "Release";

        /// <summary>
        /// Whether to keep the temporary directory after completion
        /// </summary>
        public bool KeepTempDirectory { get; set; } = false;

        /// <summary>
        /// Base directory for temporary folders (defaults to system temp)
        /// </summary>
        public string? TempBaseDirectory { get; set; }

        /// <summary>
        /// Whether to perform the build step
        /// </summary>
        public bool BuildAssembly { get; set; } = true;
    }

    /// <summary>
    /// Factory service for generating complete Swagger client projects
    /// </summary>
    public class GenSwaggerClientWorkflow
    {
        private readonly SwaggerClientCodeGenerator _codeGenerator;
        private readonly SwaggerClientCsprojCodeGenerator _csprojGenerator;
        private readonly CsprojService _csprojService;

        /// <summary>
        /// Initializes a new instance of GenSwaggerClientFactory
        /// </summary>
        public GenSwaggerClientWorkflow()
        {
            _codeGenerator = new SwaggerClientCodeGenerator();
            _csprojGenerator = new SwaggerClientCsprojCodeGenerator();
            _csprojService = new CsprojService();
        }

        /// <summary>
        /// Generates and builds a complete Swagger client project
        /// </summary>
        /// <param name="sdkName">Name of the SDK to generate</param>
        /// <param name="apiInfo">Swagger API information</param>
        /// <param name="config">Generation configuration</param>
        /// <returns>Generation and build result</returns>
        public async Task<GenSwaggerClientResult> Build(string sdkName, SwaggerApiInfo apiInfo, GenSwaggerClientConfig? config = null)
        {
            var result = new GenSwaggerClientResult
            {
                SdkName = sdkName
            };

            var startTime = DateTime.Now;
            config ??= new GenSwaggerClientConfig();

            try
            {
                // Step 1: Create temporary directory
                result.ProcessLog.Add("🔄 Step 1: Creating temporary directory...");
                result.TempDirectory = CreateTempDirectory(config.TempBaseDirectory);
                result.ProcessLog.Add($"📁 Created temp directory: {result.TempDirectory}");

                // Step 2: Generate client code
                result.ProcessLog.Add("🔄 Step 2: Generating client code...");
                result.GeneratedCode = _codeGenerator.Generate(sdkName, apiInfo);
                result.ClientCodePath = Path.Combine(result.TempDirectory, $"{sdkName}Client.cs");
                
                await File.WriteAllTextAsync(result.ClientCodePath, result.GeneratedCode);
                result.ProcessLog.Add($"✅ Generated client code: {result.ClientCodePath}");

                // Step 3: Generate .csproj file
                result.ProcessLog.Add("🔄 Step 3: Generating .csproj file...");
                var csprojConfig = new CsprojGenerationConfig
                {
                    SdkName = sdkName,
                    DotnetVersion = config.DotnetVersion,
                    OutputPath = result.TempDirectory,
                    SdkVersion = config.SdkVersion
                };

                _csprojGenerator.Generate(csprojConfig);
                result.CsprojPath = Path.Combine(result.TempDirectory, $"{sdkName}.csproj");
                result.ProcessLog.Add($"✅ Generated .csproj: {result.CsprojPath}");

                // Step 4: Build the project (if requested)
                if (config.BuildAssembly)
                {
                    result.ProcessLog.Add("🔄 Step 4: Building project...");
                    var buildConfig = new BuildConfig
                    {
                        Configuration = config.BuildConfiguration,
                        Verbosity = "minimal"
                    };

                    result.BuildResult = await _csprojService.Build(result.CsprojPath, buildConfig);
                    
                    if (result.BuildResult.Success)
                    {
                        result.AssemblyPath = result.BuildResult.AssemblyPath;
                        result.ProcessLog.Add($"✅ Build successful: {result.AssemblyPath}");
                    }
                    else
                    {
                        result.Errors.AddRange(result.BuildResult.Errors);
                        result.ProcessLog.Add("❌ Build failed");
                        foreach (var error in result.BuildResult.Errors)
                        {
                            result.ProcessLog.Add($"   {error}");
                        }
                    }
                }
                else
                {
                    result.ProcessLog.Add("⏭️  Step 4: Skipped build (BuildAssembly = false)");
                }

                // Determine overall success
                result.Success = config.BuildAssembly ? 
                    (result.BuildResult?.Success ?? false) : 
                    File.Exists(result.ClientCodePath) && File.Exists(result.CsprojPath);

                result.TotalDuration = DateTime.Now - startTime;
                result.ProcessLog.Add($"🏁 Process completed in {result.TotalDuration.TotalSeconds:F2} seconds");

                // Clean up if requested and successful
                if (!config.KeepTempDirectory && result.Success)
                {
                    result.ProcessLog.Add("🧹 Cleaning up temporary directory...");
                    // Note: We'll keep the directory for now since the assembly might be needed
                    // Directory.Delete(result.TempDirectory, true);
                }

                return result;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Factory exception: {ex.Message}");
                result.ProcessLog.Add($"💥 Fatal error: {ex.Message}");
                result.TotalDuration = DateTime.Now - startTime;
                return result;
            }
        }

        /// <summary>
        /// Generates client code and project files without building
        /// </summary>
        /// <param name="sdkName">Name of the SDK to generate</param>
        /// <param name="apiInfo">Swagger API information</param>
        /// <param name="config">Generation configuration</param>
        /// <returns>Generation result</returns>
        public async Task<GenSwaggerClientResult> GenerateOnly(string sdkName, SwaggerApiInfo apiInfo, GenSwaggerClientConfig? config = null)
        {
            config ??= new GenSwaggerClientConfig();
            config.BuildAssembly = false;
            
            return await Build(sdkName, apiInfo, config);
        }

        /// <summary>
        /// Creates a random temporary directory
        /// </summary>
        /// <param name="baseDirectory">Base directory for temp folders</param>
        /// <returns>Path to the created temporary directory</returns>
        private string CreateTempDirectory(string? baseDirectory = null)
        {
            var basePath = baseDirectory ?? Path.GetTempPath();
            var randomName = $"SwaggerClient_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
            var tempPath = Path.Combine(basePath, randomName);
            
            Directory.CreateDirectory(tempPath);
            return tempPath;
        }

        /// <summary>
        /// Prints a detailed summary of the generation result
        /// </summary>
        /// <param name="result">Generation result to summarize</param>
        public void PrintSummary(GenSwaggerClientResult result)
        {
            Console.WriteLine("\n" + new string('=', 50));
            Console.WriteLine($"📦 Swagger Client Generation Summary");
            Console.WriteLine(new string('=', 50));
            
            Console.WriteLine($"SDK Name: {result.SdkName}");
            Console.WriteLine($"Success: {(result.Success ? "✅" : "❌")}");
            Console.WriteLine($"Duration: {result.TotalDuration.TotalSeconds:F2} seconds");
            Console.WriteLine($"Temp Directory: {result.TempDirectory}");
            
            if (!string.IsNullOrEmpty(result.ClientCodePath))
            {
                Console.WriteLine($"Client Code: {result.ClientCodePath}");
            }
            
            if (!string.IsNullOrEmpty(result.CsprojPath))
            {
                Console.WriteLine($"Project File: {result.CsprojPath}");
            }
            
            if (!string.IsNullOrEmpty(result.AssemblyPath))
            {
                Console.WriteLine($"Assembly: {result.AssemblyPath}");
            }

            if (result.BuildResult != null)
            {
                Console.WriteLine($"Build Result: {(result.BuildResult.Success ? "✅" : "❌")} " +
                                $"({result.BuildResult.Duration.TotalSeconds:F2}s)");
            }

            if (result.Errors.Any())
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

        /// <summary>
        /// Cleans up the temporary directory for a result
        /// </summary>
        /// <param name="result">Result containing temp directory to clean</param>
        public void CleanupTempDirectory(GenSwaggerClientResult result)
        {
            try
            {
                if (Directory.Exists(result.TempDirectory))
                {
                    Directory.Delete(result.TempDirectory, true);
                    Console.WriteLine($"🧹 Cleaned up temporary directory: {result.TempDirectory}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Warning: Could not clean up temp directory: {ex.Message}");
            }
        }
    }
}
