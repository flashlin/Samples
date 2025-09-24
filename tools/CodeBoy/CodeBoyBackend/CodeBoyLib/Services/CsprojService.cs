using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Result of a build operation
    /// </summary>
    public class BuildResult
    {
        /// <summary>
        /// Whether the build was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Build output messages
        /// </summary>
        public List<string> Output { get; set; } = new List<string>();

        /// <summary>
        /// Build errors
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Path to the generated assembly (if successful)
        /// </summary>
        public string? AssemblyPath { get; set; }

        /// <summary>
        /// Build duration
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Exit code from the build process
        /// </summary>
        public int ExitCode { get; set; }
    }

    /// <summary>
    /// Configuration for build operations
    /// </summary>
    public class BuildConfig
    {
        /// <summary>
        /// Build configuration (Debug, Release, etc.)
        /// </summary>
        public string Configuration { get; set; } = "Release";

        /// <summary>
        /// Target framework (if different from project default)
        /// </summary>
        public string? TargetFramework { get; set; }

        /// <summary>
        /// Output directory for build artifacts
        /// </summary>
        public string? OutputPath { get; set; }

        /// <summary>
        /// Additional MSBuild properties
        /// </summary>
        public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();

        /// <summary>
        /// Verbosity level for build output
        /// </summary>
        public string Verbosity { get; set; } = "minimal";

        /// <summary>
        /// Whether to restore packages before building
        /// </summary>
        public bool NoRestore { get; set; } = false;
    }

    /// <summary>
    /// Service for building .csproj files programmatically
    /// </summary>
    public class CsprojService
    {
        /// <summary>
        /// Builds a .csproj file using dotnet CLI (recommended approach)
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <param name="config">Build configuration</param>
        /// <returns>Build result</returns>
        public async Task<BuildResult> Build(string csprojPath, BuildConfig? config = null)
        {
            return await BuildWithDotnetCli(csprojPath, config ?? new BuildConfig());
        }

        /// <summary>
        /// Builds using dotnet CLI process (most reliable method)
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <param name="config">Build configuration</param>
        /// <returns>Build result</returns>
        public async Task<BuildResult> BuildWithDotnetCli(string csprojPath, BuildConfig config)
        {
            var result = new BuildResult();
            var startTime = DateTime.Now;

            try
            {
                if (!File.Exists(csprojPath))
                {
                    result.Success = false;
                    result.Errors.Add($"Project file not found: {csprojPath}");
                    return result;
                }

                var arguments = BuildDotnetArguments(csprojPath, config);
                
                Console.WriteLine($"🔨 Building project: {Path.GetFileName(csprojPath)}");
                Console.WriteLine($"📝 Command: dotnet {arguments}");

                var processInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = arguments,
                    WorkingDirectory = Path.GetDirectoryName(csprojPath) ?? Directory.GetCurrentDirectory(),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = new Process { StartInfo = processInfo };
                
                // Capture output
                var outputLines = new List<string>();
                var errorLines = new List<string>();

                process.OutputDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        outputLines.Add(args.Data);
                        Console.WriteLine($"📤 {args.Data}");
                    }
                };

                process.ErrorDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        errorLines.Add(args.Data);
                        Console.WriteLine($"❌ {args.Data}");
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await process.WaitForExitAsync();

                result.ExitCode = process.ExitCode;
                result.Success = process.ExitCode == 0;
                result.Output = outputLines;
                result.Errors = errorLines;
                result.Duration = DateTime.Now - startTime;

                if (result.Success)
                {
                    result.AssemblyPath = FindGeneratedAssembly(csprojPath, config);
                    Console.WriteLine($"✅ Build completed successfully in {result.Duration.TotalSeconds:F2} seconds");
                    if (!string.IsNullOrEmpty(result.AssemblyPath))
                    {
                        Console.WriteLine($"📦 Assembly: {result.AssemblyPath}");
                    }
                }
                else
                {
                    Console.WriteLine($"❌ Build failed with exit code {result.ExitCode}");
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Build exception: {ex.Message}");
                result.Duration = DateTime.Now - startTime;
                Console.WriteLine($"💥 Build exception: {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Alternative: Build using MSBuild API (requires Microsoft.Build NuGet package)
        /// Note: This method is commented out because it requires additional dependencies
        /// </summary>
        /*
        public BuildResult BuildWithMSBuild(string csprojPath, BuildConfig config)
        {
            // This approach requires:
            // <PackageReference Include="Microsoft.Build" Version="17.8.3" />
            // <PackageReference Include="Microsoft.Build.Locator" Version="1.5.5" />
            
            var result = new BuildResult();
            
            try
            {
                // MSBuildLocator.RegisterDefaults();
                
                // var projectCollection = new ProjectCollection();
                // var project = projectCollection.LoadProject(csprojPath);
                
                // var buildParameters = new BuildParameters(projectCollection)
                // {
                //     Loggers = new[] { new ConsoleLogger(LoggerVerbosity.Normal) }
                // };
                
                // var buildRequest = new BuildRequestData(project.CreateProjectInstance(), new[] { "Build" });
                // var buildResult = BuildManager.DefaultBuildManager.Build(buildParameters, buildRequest);
                
                // result.Success = buildResult.OverallResult == BuildResultCode.Success;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"MSBuild error: {ex.Message}");
            }
            
            return result;
        }
        */

        /// <summary>
        /// Restores NuGet packages for the project
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <returns>Build result</returns>
        public async Task<BuildResult> Restore(string csprojPath)
        {
            var result = new BuildResult();
            var startTime = DateTime.Now;

            try
            {
                var arguments = $"restore \"{csprojPath}\"";
                
                Console.WriteLine($"📦 Restoring packages for: {Path.GetFileName(csprojPath)}");

                var processInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = arguments,
                    WorkingDirectory = Path.GetDirectoryName(csprojPath) ?? Directory.GetCurrentDirectory(),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process != null)
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    var error = await process.StandardError.ReadToEndAsync();
                    
                    await process.WaitForExitAsync();

                    result.ExitCode = process.ExitCode;
                    result.Success = process.ExitCode == 0;
                    result.Output = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).ToList();
                    result.Errors = error.Split('\n', StringSplitOptions.RemoveEmptyEntries).ToList();
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Restore exception: {ex.Message}");
            }

            result.Duration = DateTime.Now - startTime;
            return result;
        }

        /// <summary>
        /// Cleans the build output
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <returns>Build result</returns>
        public async Task<BuildResult> Clean(string csprojPath)
        {
            var result = new BuildResult();
            
            try
            {
                var arguments = $"clean \"{csprojPath}\"";
                
                var processInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = arguments,
                    WorkingDirectory = Path.GetDirectoryName(csprojPath) ?? Directory.GetCurrentDirectory(),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process != null)
                {
                    await process.WaitForExitAsync();
                    result.Success = process.ExitCode == 0;
                    result.ExitCode = process.ExitCode;
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Errors.Add($"Clean exception: {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Builds command line arguments for dotnet build
        /// </summary>
        private string BuildDotnetArguments(string csprojPath, BuildConfig config)
        {
            var args = new List<string>
            {
                "build",
                $"\"{csprojPath}\"",
                $"--configuration {config.Configuration}",
                $"--verbosity {config.Verbosity}"
            };

            if (config.NoRestore)
            {
                args.Add("--no-restore");
            }

            if (!string.IsNullOrEmpty(config.TargetFramework))
            {
                args.Add($"--framework {config.TargetFramework}");
            }

            if (!string.IsNullOrEmpty(config.OutputPath))
            {
                args.Add($"--output \"{config.OutputPath}\"");
            }

            // Add custom properties
            foreach (var prop in config.Properties)
            {
                args.Add($"-p:{prop.Key}={prop.Value}");
            }

            return string.Join(" ", args);
        }

        /// <summary>
        /// Finds the generated assembly after successful build
        /// </summary>
        private string? FindGeneratedAssembly(string csprojPath, BuildConfig config)
        {
            try
            {
                var projectDir = Path.GetDirectoryName(csprojPath);
                if (projectDir == null) return null;

                var projectName = Path.GetFileNameWithoutExtension(csprojPath);
                
                // Try custom output path first
                if (!string.IsNullOrEmpty(config.OutputPath))
                {
                    var customAssembly = Path.Combine(config.OutputPath, $"{projectName}.dll");
                    if (File.Exists(customAssembly))
                        return customAssembly;
                }

                // Try standard output paths
                var possiblePaths = new[]
                {
                    Path.Combine(projectDir, "bin", config.Configuration, "net8.0", $"{projectName}.dll"),
                    Path.Combine(projectDir, "bin", config.Configuration, "net7.0", $"{projectName}.dll"),
                    Path.Combine(projectDir, "bin", config.Configuration, "net6.0", $"{projectName}.dll"),
                    Path.Combine(projectDir, "bin", config.Configuration, "netstandard2.1", $"{projectName}.dll"),
                    Path.Combine(projectDir, "bin", config.Configuration, "netstandard2.0", $"{projectName}.dll")
                };

                return possiblePaths.FirstOrDefault(File.Exists);
            }
            catch
            {
                return null;
            }
        }
    }
}
