using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Configuration for .csproj generation
    /// </summary>
    public class CsprojGenerationConfig
    {
        /// <summary>
        /// Name of the SDK (e.g., "Petstore")
        /// </summary>
        public string SdkName { get; set; } = string.Empty;

        /// <summary>
        /// Target .NET version (e.g., "net8.0", "net6.0")
        /// </summary>
        public string DotnetVersion { get; set; } = string.Empty;

        /// <summary>
        /// Output directory path
        /// </summary>
        public string OutputPath { get; set; } = string.Empty;

        /// <summary>
        /// SDK version for package metadata (e.g., "1.0.0", "2.1.0")
        /// </summary>
        public string SdkVersion { get; set; } = "1.0.0";
    }

    /// <summary>
    /// Generates .csproj files for Swagger client libraries
    /// </summary>
    public class SwaggerClientCsprojCodeGenerator
    {
        /// <summary>
        /// Generates a .csproj file for the SDK library using configuration object
        /// </summary>
        /// <param name="config">Configuration for .csproj generation</param>
        public void Generate(CsprojGenerationConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (string.IsNullOrEmpty(config.SdkName))
                throw new ArgumentException("SDK name cannot be null or empty", nameof(config));
            
            if (string.IsNullOrEmpty(config.DotnetVersion))
                throw new ArgumentException("Dotnet version cannot be null or empty", nameof(config));
            
            if (string.IsNullOrEmpty(config.OutputPath))
                throw new ArgumentException("Output path cannot be null or empty", nameof(config));

            // Create output directory if it doesn't exist
            Directory.CreateDirectory(config.OutputPath);

            // Generate .csproj content
            var csprojContent = GenerateCsprojContent(config);
            
            // Write .csproj file
            var csprojFilePath = Path.Combine(config.OutputPath, $"{config.SdkName}.csproj");
            File.WriteAllText(csprojFilePath, csprojContent, Encoding.UTF8);
            
            Console.WriteLine($"✅ Generated .csproj file: {csprojFilePath}");
        }

        /// <summary>
        /// Generates a .csproj file for the SDK library (legacy method for backward compatibility)
        /// </summary>
        /// <param name="sdkName">Name of the SDK (e.g., "Petstore")</param>
        /// <param name="dotnetVersion">Target .NET version (e.g., "net8.0", "net6.0")</param>
        /// <param name="outputPath">Output directory path</param>
        public void Generate(string sdkName, string dotnetVersion, string outputPath)
        {
            var config = new CsprojGenerationConfig
            {
                SdkName = sdkName,
                DotnetVersion = dotnetVersion,
                OutputPath = outputPath,
                SdkVersion = "1.0.0" // Default version for backward compatibility
            };
            
            Generate(config);
        }

        /// <summary>
        /// Generates the content of the .csproj file
        /// </summary>
        /// <param name="config">Configuration for .csproj generation</param>
        /// <returns>Generated .csproj XML content</returns>
        private string GenerateCsprojContent(CsprojGenerationConfig config)
        {
            var sb = new StringBuilder();
            
            // XML declaration and project start
            sb.AppendLine("<Project Sdk=\"Microsoft.NET.Sdk\">");
            sb.AppendLine();
            
            // PropertyGroup
            sb.AppendLine("  <PropertyGroup>");
            sb.AppendLine($"    <TargetFramework>{config.DotnetVersion}</TargetFramework>");
            sb.AppendLine("    <ImplicitUsings>enable</ImplicitUsings>");
            sb.AppendLine("    <Nullable>enable</Nullable>");
            sb.AppendLine($"    <AssemblyName>{config.SdkName}</AssemblyName>");
            sb.AppendLine($"    <RootNamespace>{config.SdkName}SDK</RootNamespace>");
            sb.AppendLine($"    <PackageId>{config.SdkName}.Client</PackageId>");
            sb.AppendLine("    <GeneratePackageOnBuild>true</GeneratePackageOnBuild>");
            sb.AppendLine($"    <PackageDescription>Auto-generated API client for {config.SdkName}</PackageDescription>");
            sb.AppendLine($"    <Version>{config.SdkVersion}</Version>");
            sb.AppendLine("    <Authors>CodeBoy Generator</Authors>");
            sb.AppendLine("  </PropertyGroup>");
            sb.AppendLine();

            // PackageReference - use latest secure versions
            var packageVersions = GetRecommendedPackageVersions(config.DotnetVersion);
            sb.AppendLine("  <ItemGroup>");
            foreach (var package in packageVersions)
            {
                sb.AppendLine($"    <PackageReference Include=\"{package.Key}\" Version=\"{package.Value}\" />");
            }
            sb.AppendLine("  </ItemGroup>");
            sb.AppendLine();

            // Note: .NET SDK automatically includes .cs files, so we don't need to explicitly specify them
            // This prevents NETSDK1022 error about duplicate items
            var csFiles = ScanForCsFiles(config.OutputPath);
            if (csFiles.Any())
            {
                Console.WriteLine($"📝 Note: Found {csFiles.Count} .cs files that will be automatically included by the .NET SDK");
            }

            // Project end
            sb.AppendLine("</Project>");

            return sb.ToString();
        }

        /// <summary>
        /// Scans the output directory for .cs files
        /// </summary>
        /// <param name="outputPath">Directory to scan</param>
        /// <returns>List of .cs file paths</returns>
        private List<string> ScanForCsFiles(string outputPath)
        {
            var csFiles = new List<string>();
            
            try
            {
                if (Directory.Exists(outputPath))
                {
                    // Get all .cs files in the directory and subdirectories
                    var files = Directory.GetFiles(outputPath, "*.cs", SearchOption.AllDirectories);
                    csFiles.AddRange(files);
                    
                    Console.WriteLine($"📁 Found {csFiles.Count} .cs files in {outputPath}");
                    foreach (var file in csFiles)
                    {
                        var relativePath = Path.GetRelativePath(outputPath, file);
                        Console.WriteLine($"   - {relativePath}");
                    }
                }
                else
                {
                    Console.WriteLine($"📁 Directory {outputPath} does not exist yet - will create empty project");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Warning: Could not scan for .cs files in {outputPath}: {ex.Message}");
            }
            
            return csFiles;
        }

        /// <summary>
        /// Validates if the dotnet version format is correct
        /// </summary>
        /// <param name="dotnetVersion">Version string to validate</param>
        /// <returns>True if valid format</returns>
        public bool IsValidDotnetVersion(string dotnetVersion)
        {
            if (string.IsNullOrEmpty(dotnetVersion))
                return false;

            // Common valid formats: net8.0, net6.0, net7.0, netstandard2.0, netstandard2.1
            var validPatterns = new[]
            {
                "net6.0", "net7.0", "net8.0", "net9.0",
                "netstandard2.0", "netstandard2.1",
                "netcoreapp3.1"
            };

            return validPatterns.Contains(dotnetVersion.ToLowerInvariant()) ||
                   dotnetVersion.StartsWith("net") && dotnetVersion.Contains(".");
        }

        /// <summary>
        /// Gets recommended package versions based on target framework
        /// </summary>
        /// <param name="dotnetVersion">Target .NET version</param>
        /// <returns>Dictionary of package name to version</returns>
        private Dictionary<string, string> GetRecommendedPackageVersions(string dotnetVersion)
        {
            var versions = new Dictionary<string, string>();

            if (dotnetVersion.StartsWith("net9"))
            {
                versions["Microsoft.Extensions.Http"] = "9.0.0";
                versions["Microsoft.Extensions.Options"] = "9.0.0";
                versions["System.Text.Json"] = "9.0.0";
                versions["System.ComponentModel.Annotations"] = "5.0.0";
            }
            else if (dotnetVersion.StartsWith("net8"))
            {
                versions["Microsoft.Extensions.Http"] = "8.0.1";
                versions["Microsoft.Extensions.Options"] = "8.0.2";
                versions["System.Text.Json"] = "8.0.5"; // Latest secure version
                versions["System.ComponentModel.Annotations"] = "5.0.0";
            }
            else if (dotnetVersion.StartsWith("net7"))
            {
                versions["Microsoft.Extensions.Http"] = "7.0.0";
                versions["Microsoft.Extensions.Options"] = "7.0.0";
                versions["System.Text.Json"] = "7.0.0";
                versions["System.ComponentModel.Annotations"] = "5.0.0";
            }
            else if (dotnetVersion.StartsWith("net6"))
            {
                versions["Microsoft.Extensions.Http"] = "6.0.0";
                versions["Microsoft.Extensions.Options"] = "6.0.0";
                versions["System.Text.Json"] = "6.0.0";
                versions["System.ComponentModel.Annotations"] = "5.0.0";
            }
            else
            {
                // Default to compatible versions
                versions["Microsoft.Extensions.Http"] = "6.0.0";
                versions["Microsoft.Extensions.Options"] = "6.0.0";
                versions["System.Text.Json"] = "6.0.0";
                versions["System.ComponentModel.Annotations"] = "5.0.0";
            }

            return versions;
        }
    }
}
