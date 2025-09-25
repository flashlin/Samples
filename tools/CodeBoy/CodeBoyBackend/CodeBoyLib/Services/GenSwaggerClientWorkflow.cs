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
        public bool KeepTempDirectory { get; set; } = true;

        /// <summary>
        /// Base directory for temporary folders (defaults to system temp)
        /// </summary>
        public required string TempBaseDirectory { get; set; }

        /// <summary>
        /// Whether to perform the build step
        /// </summary>
        public bool BuildAssembly { get; set; } = true;
    }

    /// <summary>
    /// Parameters for building a Swagger client
    /// </summary>
    public class GenSwaggerClientBuildParams
    {
        /// <summary>
        /// Name of the SDK to generate
        /// </summary>
        public required string SdkName { get; set; }

        /// <summary>
        /// Swagger API information
        /// </summary>
        public required SwaggerApiInfo ApiInfo { get; set; }

        /// <summary>
        /// Base output path for all build operations
        /// </summary>
        public required string OutputPath { get; set; }

        /// <summary>
        /// NuGet package name
        /// </summary>
        public required string NupkgName { get; set; }

        /// <summary>
        /// SDK version for the generated package
        /// </summary>
        public string SdkVersion { get; set; } = "1.0.0";
    }

    /// <summary>
    /// Factory service for generating complete Swagger client projects
    /// </summary>
    public class GenSwaggerClientWorkflow
    {
        private readonly SwaggerClientCodeGenerator _codeGenerator;
        private readonly SwaggerClientCsprojCodeGenerator _csprojGenerator;
        private readonly CsprojService _csprojService;
        private readonly NupkgFileGenerator _nupkgGenerator;

        /// <summary>
        /// Initializes a new instance of GenSwaggerClientWorkflow
        /// </summary>
        public GenSwaggerClientWorkflow()
        {
            _codeGenerator = new SwaggerClientCodeGenerator();
            _csprojGenerator = new SwaggerClientCsprojCodeGenerator();
            _csprojService = new CsprojService();
            _nupkgGenerator = new NupkgFileGenerator();
        }

        /// <summary>
        /// Generates and builds a complete Swagger client project for multiple target frameworks
        /// </summary>
        /// <param name="buildParams">Build parameters</param>
        /// <returns>Generation and build result</returns>
        public async Task<GenSwaggerClientResult> Build(GenSwaggerClientBuildParams buildParams)
        {
            var result = new GenSwaggerClientResult
            {
                SdkName = buildParams.SdkName
            };

            var startTime = DateTime.Now;
            var targetFrameworks = new[] { "net8.0", "net9.0" };
            var outputPathList = new List<string>();

            try
            {
                result.ProcessLog.Add($"🚀 Starting multi-target build for frameworks: {string.Join(", ", targetFrameworks)}");
                result.ProcessLog.Add($"📁 Using output path: {buildParams.OutputPath}");

                // Ensure output directory exists
                Directory.CreateDirectory(buildParams.OutputPath);

                // Build for each target framework
                await BuildAllFrameworks(buildParams.SdkName, buildParams.ApiInfo, targetFrameworks, result, outputPathList, buildParams.OutputPath, buildParams.SdkVersion);

                // Generate NuGet package if any frameworks were successful
                await GenerateNuGetPackage(buildParams.NupkgName, outputPathList, result, buildParams.OutputPath, buildParams.SdkVersion);

                // Finalize the multi-target build result
                FinalizeMultiTargetResult(result, outputPathList, targetFrameworks, startTime);

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
        /// Builds all target frameworks sequentially
        /// </summary>
        /// <param name="sdkName">Name of the SDK</param>
        /// <param name="apiInfo">Swagger API information</param>
        /// <param name="targetFrameworks">Array of target frameworks to build</param>
        /// <param name="result">Main result object to update</param>
        /// <param name="outputPathList">List to collect successful output paths</param>
        /// <param name="outputPath">Base output path for all build operations</param>
        /// <param name="sdkVersion"></param>
        private async Task BuildAllFrameworks(string sdkName, SwaggerApiInfo apiInfo, string[] targetFrameworks, 
            GenSwaggerClientResult result, List<string> outputPathList, string outputPath, string sdkVersion)
        {
            foreach (var framework in targetFrameworks)
            {
                result.ProcessLog.Add($"🔄 Building for {framework}...");
                
                var frameworkSuccess = await BuildSingleFramework(sdkName, apiInfo, framework, result, outputPathList, outputPath, sdkVersion);
                
                result.ProcessLog.Add($"{(frameworkSuccess ? "✅" : "❌")} {framework} build {(frameworkSuccess ? "successful" : "failed")}");
            }
        }

        /// <summary>
        /// Builds a single target framework
        /// </summary>
        /// <param name="sdkName">Name of the SDK</param>
        /// <param name="apiInfo">Swagger API information</param>
        /// <param name="framework">Target framework (e.g., "net8.0")</param>
        /// <param name="result">Main result object to update</param>
        /// <param name="outputPathList">List to collect successful output paths</param>
        /// <param name="outputPath">Base output path for build operations</param>
        /// <param name="sdkVersion"></param>
        /// <returns>True if successful, false otherwise</returns>
        private async Task<bool> BuildSingleFramework(string sdkName, SwaggerApiInfo apiInfo, string framework, 
            GenSwaggerClientResult result, List<string> outputPathList, string outputPath, string sdkVersion)
        {
            try
            {
                // Create framework-specific config with outputPath-based TempBaseDirectory
                var config = CreateFrameworkConfig(framework, outputPath, sdkVersion);

                // Create a temporary result object for this framework
                var frameworkResult = new GenSwaggerClientResult
                {
                    SdkName = sdkName
                };

                // Execute workflow steps for this framework
                CreateTempDirectoryStep(frameworkResult, config);
                GenerateCsprojStep(frameworkResult, sdkName, config);
                await GenerateClientCodeStep(frameworkResult, sdkName, apiInfo, frameworkResult.TempDirectory);
                await BuildProjectStep(frameworkResult, config);

                // Check if framework build was successful
                var frameworkSuccess = config.BuildAssembly ? 
                    (frameworkResult.BuildResult?.Success ?? false) : 
                    File.Exists(frameworkResult.ClientCodePath) && File.Exists(frameworkResult.CsprojPath);

                if (frameworkSuccess)
                {
                    outputPathList.Add(frameworkResult.TempDirectory);
                    
                    // For the first successful framework, copy paths to main result
                    if (string.IsNullOrEmpty(result.TempDirectory))
                    {
                        CopyFrameworkResultToMain(frameworkResult, result);
                    }
                }
                else
                {
                    result.Errors.AddRange(frameworkResult.Errors);
                }

                // Add framework-specific logs to main result
                result.ProcessLog.AddRange(frameworkResult.ProcessLog);

                return frameworkSuccess;
            }
            catch (Exception ex)
            {
                result.ProcessLog.Add($"❌ {framework} build failed with exception: {ex.Message}");
                result.Errors.Add($"{framework} build exception: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Creates a framework-specific configuration
        /// </summary>
        /// <param name="framework">Target framework</param>
        /// <param name="outputPath">Base output path for build operations</param>
        /// <param name="sdkVersion"></param>
        /// <returns>Configuration for the framework</returns>
        private GenSwaggerClientConfig CreateFrameworkConfig(string framework, string outputPath, string sdkVersion)
        {
            return new GenSwaggerClientConfig
            {
                DotnetVersion = framework,
                SdkVersion = sdkVersion,
                BuildConfiguration = "Release",
                KeepTempDirectory = true,
                TempBaseDirectory = outputPath,
                BuildAssembly = true
            };
        }

        /// <summary>
        /// Copies framework-specific results to the main result object
        /// </summary>
        /// <param name="frameworkResult">Framework-specific result</param>
        /// <param name="mainResult">Main result object</param>
        private void CopyFrameworkResultToMain(GenSwaggerClientResult frameworkResult, GenSwaggerClientResult mainResult)
        {
            mainResult.TempDirectory = frameworkResult.TempDirectory;
            mainResult.ClientCodePath = frameworkResult.ClientCodePath;
            mainResult.CsprojPath = frameworkResult.CsprojPath;
            mainResult.AssemblyPath = frameworkResult.AssemblyPath;
            mainResult.GeneratedCode = frameworkResult.GeneratedCode;
            mainResult.BuildResult = frameworkResult.BuildResult;
        }

        private async Task GenerateNuGetPackage(string nupkgName, List<string> outputPathList, GenSwaggerClientResult result, string outputPath, string sdkVersion)
        {
            if (outputPathList.Count > 0)
            {
                result.ProcessLog.Add("🔄 Generating NuGet package...");
                var nupkgFile = Path.Combine(outputPath, $"{nupkgName}.{sdkVersion}.nupkg");
                
                var nupkgSuccess = _nupkgGenerator.Generate(nupkgFile, outputPathList, sdkVersion);
                if (nupkgSuccess)
                {
                    result.ProcessLog.Add($"✅ NuGet package created: {nupkgFile}");
                }
                else
                {
                    result.ProcessLog.Add("❌ Failed to create NuGet package");
                    result.Errors.Add("NuGet package generation failed");
                }
            }
            await Task.CompletedTask; // Make method async for consistency
        }

        /// <summary>
        /// Finalizes the multi-target build result
        /// </summary>
        /// <param name="result">Result object to finalize</param>
        /// <param name="outputPathList">List of successful output paths</param>
        /// <param name="targetFrameworks">Array of target frameworks that were attempted</param>
        /// <param name="startTime">Start time of the build process</param>
        private void FinalizeMultiTargetResult(GenSwaggerClientResult result, List<string> outputPathList, 
            string[] targetFrameworks, DateTime startTime)
        {
            // Determine overall success
            result.Success = outputPathList.Count > 0;
            result.TotalDuration = DateTime.Now - startTime;
            result.ProcessLog.Add($"🏁 Multi-target build completed in {result.TotalDuration.TotalSeconds:F2} seconds");
            result.ProcessLog.Add($"📊 Successful frameworks: {outputPathList.Count}/{targetFrameworks.Length}");
        }

        /// <summary>
        /// Step 1: Create temporary directory for the generation process
        /// </summary>
        /// <param name="result">Result object to update</param>
        /// <param name="config">Generation configuration</param>
        private void CreateTempDirectoryStep(GenSwaggerClientResult result, GenSwaggerClientConfig config)
        {
            result.ProcessLog.Add("🔄 Step 1: Creating temporary directory...");
            result.TempDirectory = CreateTempDirectory(config.TempBaseDirectory);
            result.ProcessLog.Add($"📁 Created temp directory: {result.TempDirectory}");
        }

        /// <summary>
        /// Step 2: Generate client code and save to file
        /// </summary>
        /// <param name="result">Result object to update</param>
        /// <param name="sdkName">Name of the SDK</param>
        /// <param name="apiInfo">Swagger API information</param>
        /// <param name="frameworkTempDirectory">Framework-specific temp directory for file generation</param>
        private async Task GenerateClientCodeStep(GenSwaggerClientResult result, string sdkName, SwaggerApiInfo apiInfo, string frameworkTempDirectory)
        {
            result.ProcessLog.Add("🔄 Step 2: Generating client code...");
            result.GeneratedCode = _codeGenerator.Generate(sdkName, apiInfo);
            result.ClientCodePath = Path.Combine(frameworkTempDirectory, $"{sdkName}Client.cs");
            
            await File.WriteAllTextAsync(result.ClientCodePath, result.GeneratedCode);
            result.ProcessLog.Add($"✅ Generated client code: {result.ClientCodePath}");
        }

        /// <summary>
        /// Step 3: Generate .csproj file
        /// </summary>
        /// <param name="result">Result object to update</param>
        /// <param name="sdkName">Name of the SDK</param>
        /// <param name="config">Generation configuration</param>
        private void GenerateCsprojStep(GenSwaggerClientResult result, string sdkName, GenSwaggerClientConfig config)
        {
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
        }

        /// <summary>
        /// Step 4: Build the project (if requested)
        /// </summary>
        /// <param name="result">Result object to update</param>
        /// <param name="config">Generation configuration</param>
        private async Task BuildProjectStep(GenSwaggerClientResult result, GenSwaggerClientConfig config)
        {
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
        }

        /// <summary>
        /// Finalize the result and perform cleanup
        /// </summary>
        /// <param name="result">Result object to finalize</param>
        /// <param name="config">Generation configuration</param>
        /// <param name="startTime">Start time of the process</param>
        private void FinalizeResult(GenSwaggerClientResult result, GenSwaggerClientConfig config, DateTime startTime)
        {
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
                Directory.Delete(result.TempDirectory, true);
            }
        }

        /// <summary>
        /// Creates a random temporary directory
        /// </summary>
        /// <param name="baseDirectory">Base directory for temp folders</param>
        /// <returns>Absolute path to the created temporary directory</returns>
        private string CreateTempDirectory(string? baseDirectory = null)
        {
            var basePath = baseDirectory ?? Path.GetTempPath();
            var randomName = $"SwaggerClient_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
            var tempPath = Path.Combine(basePath, randomName);
            
            Directory.CreateDirectory(tempPath);
            
            // Ensure we return an absolute path
            return Path.GetFullPath(tempPath);
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
