using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Information about a project dependency
    /// </summary>
    public class ProjectDependency
    {
        /// <summary>
        /// Package or assembly name
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Version of the dependency (for NuGet packages)
        /// </summary>
        public string? Version { get; set; }

        /// <summary>
        /// Type of dependency (PackageReference, Reference, etc.)
        /// </summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>
        /// Target framework for this dependency (if specific)
        /// </summary>
        public string? TargetFramework { get; set; }
    }

    /// <summary>
    /// Information about a project to be included in the NuGet package
    /// </summary>
    public class ProjectInfo
    {
        /// <summary>
        /// Path to the .csproj file
        /// </summary>
        public string CsprojPath { get; set; } = string.Empty;

        /// <summary>
        /// Target framework (e.g., net8.0, net9.0)
        /// </summary>
        public string TargetFramework { get; set; } = string.Empty;

        /// <summary>
        /// Output directory path
        /// </summary>
        public string OutputPath { get; set; } = string.Empty;

        /// <summary>
        /// Project name
        /// </summary>
        public string ProjectName { get; set; } = string.Empty;

        /// <summary>
        /// Dependencies extracted from the .csproj file
        /// </summary>
        public List<ProjectDependency> Dependencies { get; set; } = new List<ProjectDependency>();
    }

    /// <summary>
    /// Service for generating NuGet package (.nupkg) files
    /// </summary>
    public class NupkgFileGenerator
    {
        /// <summary>
        /// Generates a .nupkg file from multiple project output paths
        /// </summary>
        /// <param name="nupkgFile">Path where the .nupkg file should be created</param>
        /// <param name="csProjectPathList">List of output directory paths containing .csproj files</param>
        /// <param name="sdkVersion"></param>
        /// <returns>True if successful, false otherwise</returns>
        public bool Generate(string nupkgFile, List<string> csProjectPathList, string sdkVersion)
        {
            try
            {
                Console.WriteLine($"🚀 Starting NuGet package generation: {Path.GetFileName(nupkgFile)}");

                // Create tmp directory in Generated folder
                var generatedDir = Path.Combine(Directory.GetCurrentDirectory(), "Generated");
                var tmpDir = Path.Combine(generatedDir, "tmp");
                
                if (Directory.Exists(tmpDir))
                {
                    Directory.Delete(tmpDir, true);
                }
                Directory.CreateDirectory(tmpDir);
                
                Console.WriteLine($"📁 Created temporary directory: {tmpDir}");

                // Collect project information from all output paths
                var projects = new List<ProjectInfo>();
                foreach (var outputPath in csProjectPathList)
                {
                    var projectInfo = AnalyzeOutputPath(outputPath);
                    if (projectInfo != null)
                    {
                        projects.Add(projectInfo);
                        Console.WriteLine($"📋 Found project: {projectInfo.ProjectName} ({projectInfo.TargetFramework})");
                    }
                }

                if (projects.Count == 0)
                {
                    Console.WriteLine("❌ No valid projects found in the provided output paths");
                    return false;
                }

                // Create the package structure in tmp directory
                var success = CreateNupkgStructure(tmpDir, projects, nupkgFile, sdkVersion);
                
                if (success)
                {
                    Console.WriteLine($"✅ Successfully created NuGet package: {nupkgFile}");
                }
                else
                {
                    Console.WriteLine("❌ Failed to create NuGet package");
                }

                // Clean up tmp directory
                try
                {
                    Directory.Delete(tmpDir, true);
                    Console.WriteLine("🧹 Cleaned up temporary directory");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Warning: Could not clean up tmp directory: {ex.Message}");
                }

                return success;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"💥 Error generating NuGet package: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Analyzes an output path to find .csproj file and extract project information
        /// </summary>
        /// <param name="outputPath">Path to analyze</param>
        /// <returns>ProjectInfo if successful, null otherwise</returns>
        private ProjectInfo? AnalyzeOutputPath(string outputPath)
        {
            try
            {
                if (!Directory.Exists(outputPath))
                {
                    Console.WriteLine($"⚠️ Output path does not exist: {outputPath}");
                    return null;
                }

                // Look for .csproj file in the output path
                var csprojFiles = Directory.GetFiles(outputPath, "*.csproj", SearchOption.TopDirectoryOnly);
                
                if (csprojFiles.Length == 0)
                {
                    Console.WriteLine($"⚠️ No .csproj file found in: {outputPath}");
                    return null;
                }

                if (csprojFiles.Length > 1)
                {
                    Console.WriteLine($"⚠️ Multiple .csproj files found in {outputPath}, using the first one");
                }

                var csprojPath = csprojFiles[0];
                var targetFramework = ExtractTargetFramework(csprojPath);
                
                if (string.IsNullOrEmpty(targetFramework))
                {
                    Console.WriteLine($"⚠️ Could not extract TargetFramework from: {csprojPath}");
                    return null;
                }

                var dependencies = ExtractDependenciesFromCsproj(csprojPath);

                return new ProjectInfo
                {
                    CsprojPath = csprojPath,
                    TargetFramework = targetFramework,
                    OutputPath = outputPath,
                    ProjectName = Path.GetFileNameWithoutExtension(csprojPath),
                    Dependencies = dependencies
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error analyzing output path {outputPath}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Extracts TargetFramework from a .csproj file
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <returns>Target framework string (e.g., "net8.0") or empty string if not found</returns>
        private string ExtractTargetFramework(string csprojPath)
        {
            try
            {
                var doc = XDocument.Load(csprojPath);
                var targetFramework = doc.Descendants("TargetFramework").FirstOrDefault()?.Value;
                
                if (!string.IsNullOrEmpty(targetFramework))
                {
                    return targetFramework.Trim();
                }

                // Also check for TargetFrameworks (plural) in case of multi-targeting
                var targetFrameworks = doc.Descendants("TargetFrameworks").FirstOrDefault()?.Value;
                if (!string.IsNullOrEmpty(targetFrameworks))
                {
                    // Take the first framework if multiple are specified
                    return targetFrameworks.Split(';').FirstOrDefault()?.Trim() ?? string.Empty;
                }

                return string.Empty;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error reading .csproj file {csprojPath}: {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Extracts dependencies from a .csproj file
        /// </summary>
        /// <param name="csprojPath">Path to the .csproj file</param>
        /// <returns>List of project dependencies</returns>
        private List<ProjectDependency> ExtractDependenciesFromCsproj(string csprojPath)
        {
            var dependencies = new List<ProjectDependency>();

            try
            {
                var doc = XDocument.Load(csprojPath);

                // Extract PackageReference dependencies
                var packageReferences = doc.Descendants("PackageReference");
                foreach (var packageRef in packageReferences)
                {
                    var name = packageRef.Attribute("Include")?.Value;
                    var version = packageRef.Attribute("Version")?.Value;
                    
                    if (!string.IsNullOrEmpty(name))
                    {
                        dependencies.Add(new ProjectDependency
                        {
                            Name = name,
                            Version = version,
                            Type = "PackageReference"
                        });
                    }
                }

                // Extract Reference dependencies (assembly references)
                var references = doc.Descendants("Reference");
                foreach (var reference in references)
                {
                    var name = reference.Attribute("Include")?.Value;
                    
                    if (!string.IsNullOrEmpty(name))
                    {
                        // Extract just the assembly name (before comma if version info is present)
                        var assemblyName = name.Split(',')[0].Trim();
                        
                        dependencies.Add(new ProjectDependency
                        {
                            Name = assemblyName,
                            Type = "Reference"
                        });
                    }
                }

                // Extract ProjectReference dependencies
                var projectReferences = doc.Descendants("ProjectReference");
                foreach (var projectRef in projectReferences)
                {
                    var includePath = projectRef.Attribute("Include")?.Value;
                    
                    if (!string.IsNullOrEmpty(includePath))
                    {
                        var projectName = Path.GetFileNameWithoutExtension(includePath);
                        
                        dependencies.Add(new ProjectDependency
                        {
                            Name = projectName,
                            Type = "ProjectReference"
                        });
                    }
                }

                Console.WriteLine($"📦 Extracted {dependencies.Count} dependencies from {Path.GetFileName(csprojPath)}");
                foreach (var dep in dependencies)
                {
                    var versionInfo = !string.IsNullOrEmpty(dep.Version) ? $" (v{dep.Version})" : "";
                    Console.WriteLine($"   - {dep.Type}: {dep.Name}{versionInfo}");
                }

                return dependencies;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error extracting dependencies from {csprojPath}: {ex.Message}");
                return dependencies;
            }
        }

        /// <summary>
        /// Creates the NuGet package structure and generates the .nupkg file
        /// </summary>
        /// <param name="tmpDir">Temporary directory for package creation</param>
        /// <param name="projects">List of projects to include</param>
        /// <param name="nupkgFile">Output .nupkg file path</param>
        /// <param name="sdkVersion"></param>
        /// <returns>True if successful</returns>
        private bool CreateNupkgStructure(string tmpDir, List<ProjectInfo> projects, string nupkgFile, string sdkVersion)
        {
            try
            {
                // Create lib directory structure for each target framework
                foreach (var project in projects)
                {
                    var libDir = Path.Combine(tmpDir, "lib", project.TargetFramework);
                    Directory.CreateDirectory(libDir);

                    // Copy assemblies from bin/Debug/{framework}/* or bin/Release/{framework}/*
                    var debugBinDir = Path.Combine(project.OutputPath, "bin", "Debug", project.TargetFramework);
                    var releaseBinDir = Path.Combine(project.OutputPath, "bin", "Release", project.TargetFramework);
                    
                    string sourceBinDir;
                    if (Directory.Exists(releaseBinDir))
                    {
                        sourceBinDir = releaseBinDir;
                        Console.WriteLine($"📦 Using Release build for {project.ProjectName}");
                    }
                    else if (Directory.Exists(debugBinDir))
                    {
                        sourceBinDir = debugBinDir;
                        Console.WriteLine($"📦 Using Debug build for {project.ProjectName}");
                    }
                    else
                    {
                        Console.WriteLine($"⚠️ No bin directory found for {project.ProjectName} in {project.OutputPath}");
                        continue;
                    }

                    // Copy all files from the bin directory
                    foreach (var file in Directory.GetFiles(sourceBinDir, "*", SearchOption.AllDirectories))
                    {
                        var relativePath = Path.GetRelativePath(sourceBinDir, file);
                        var destPath = Path.Combine(libDir, relativePath);
                        
                        var destDir = Path.GetDirectoryName(destPath);
                        if (!string.IsNullOrEmpty(destDir))
                        {
                            Directory.CreateDirectory(destDir);
                        }
                        
                        File.Copy(file, destPath, true);
                    }

                    Console.WriteLine($"📋 Copied assemblies for {project.TargetFramework} to package");
                }

                // Create .nuspec file
                var nuspecPath = Path.Combine(tmpDir, "package.nuspec");
                CreateNuspecFile(nuspecPath, projects, sdkVersion);

                // Create the .nupkg file
                CreateNupkgFile(tmpDir, nupkgFile);

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating package structure: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Generates the files section for the .nuspec file
        /// </summary>
        /// <param name="projects">List of projects to include in the package</param>
        /// <returns>Generated files XML section</returns>
        private string GenerateFilesSection(List<ProjectInfo> projects)
        {
            var filesSection = new StringBuilder();
            filesSection.AppendLine("  <files>");

            foreach (var project in projects)
            {
                var framework = project.TargetFramework;
                var projectName = project.ProjectName;
                
                filesSection.AppendLine($"    <!-- {framework} -->");
                filesSection.AppendLine($"    <file src=\"lib/{framework}/{projectName}.dll\" target=\"lib/{framework}\" />");
                filesSection.AppendLine($"    <file src=\"lib/{framework}/{projectName}.pdb\" target=\"lib/{framework}\" />");
                filesSection.AppendLine($"    <file src=\"lib/{framework}/{projectName}.deps.json\" target=\"lib/{framework}\" />");
                filesSection.AppendLine();
            }

            filesSection.AppendLine("  </files>");
            return filesSection.ToString();
        }

        /// <summary>
        /// Generates the dependencies section for the .nuspec file
        /// </summary>
        /// <param name="projects">List of projects to analyze for dependencies</param>
        /// <returns>Generated dependencies XML section</returns>
        private string GenerateDependenciesSection(List<ProjectInfo> projects)
        {
            var dependenciesSection = new StringBuilder();
            
            // Collect all PackageReference dependencies from all projects
            var allDependencies = new Dictionary<string, (string? Version, string TargetFramework)>();
            
            foreach (var project in projects)
            {
                var packageDependencies = project.Dependencies
                    .Where(d => d.Type == "PackageReference")
                    .ToList();
                
                foreach (var dependency in packageDependencies)
                {
                    var key = $"{dependency.Name}_{project.TargetFramework}";
                    if (!allDependencies.ContainsKey(key))
                    {
                        allDependencies[key] = (dependency.Version, project.TargetFramework);
                    }
                }
            }
            
            if (allDependencies.Count > 0)
            {
                dependenciesSection.AppendLine("    <dependencies>");
                
                // Group by target framework
                var frameworkGroups = allDependencies
                    .GroupBy(kvp => kvp.Value.TargetFramework)
                    .OrderBy(g => g.Key);
                
                foreach (var frameworkGroup in frameworkGroups)
                {
                    dependenciesSection.AppendLine($"      <group targetFramework=\"{frameworkGroup.Key}\">");
                    
                    foreach (var dependency in frameworkGroup.OrderBy(d => d.Key))
                    {
                        var packageName = dependency.Key.Substring(0, dependency.Key.LastIndexOf('_'));
                        var version = dependency.Value.Version ?? "1.0.0";
                        
                        dependenciesSection.AppendLine($"        <dependency id=\"{packageName}\" version=\"{version}\" />");
                    }
                    
                    dependenciesSection.AppendLine("      </group>");
                }
                
                dependenciesSection.AppendLine("    </dependencies>");
            }
            
            return dependenciesSection.ToString();
        }

        /// <summary>
        /// Creates the .nuspec manifest file
        /// </summary>
        /// <param name="nuspecPath">Path where to create the .nuspec file</param>
        /// <param name="projects">List of projects to include in the package</param>
        /// <param name="sdkVersion"></param>
        private void CreateNuspecFile(string nuspecPath, List<ProjectInfo> projects, string sdkVersion)
        {
            var primaryProject = projects.FirstOrDefault();
            if (primaryProject == null) return;

            var frameworks = string.Join(", ", projects.Select(p => p.TargetFramework).Distinct());
            var projectNames = string.Join(", ", projects.Select(p => p.ProjectName).Distinct());

            var filesSection = GenerateFilesSection(projects);
            var dependenciesSection = GenerateDependenciesSection(projects);

            var nuspecContent = $@"<?xml version=""1.0"" encoding=""utf-8""?>
<package xmlns=""http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd"">
  <metadata>
    <id>{primaryProject.ProjectName}</id>
    <version>{sdkVersion}</version>
    <title>{primaryProject.ProjectName} Client</title>
    <authors>CodeBoy Generator</authors>
    <description>Auto-generated client library for {projectNames}. Supports {frameworks}.</description>
    <projectUrl>https://github.com/codeboy</projectUrl>
    <licenseUrl>https://opensource.org/licenses/MIT</licenseUrl>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <tags>api client codegen swagger openapi</tags>
{dependenciesSection}  </metadata>
{filesSection}</package>";

            File.WriteAllText(nuspecPath, nuspecContent);
            Console.WriteLine($"📝 Created .nuspec file: {nuspecPath}");
        }

        /// <summary>
        /// Creates the final .nupkg file from the package structure
        /// </summary>
        /// <param name="tmpDir">Directory containing the package structure</param>
        /// <param name="nupkgFile">Output .nupkg file path</param>
        private void CreateNupkgFile(string tmpDir, string nupkgFile)
        {
            // Ensure the output directory exists
            var outputDir = Path.GetDirectoryName(nupkgFile);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Create the zip file (which is what a .nupkg file actually is)
            using (var archive = ZipFile.Open(nupkgFile, ZipArchiveMode.Create))
            {
                AddDirectoryToZip(archive, tmpDir, string.Empty);
            }

            Console.WriteLine($"📦 Created NuGet package: {nupkgFile}");
        }

        /// <summary>
        /// Recursively adds directory contents to a zip archive
        /// </summary>
        /// <param name="archive">Target zip archive</param>
        /// <param name="sourceDir">Source directory to add</param>
        /// <param name="entryPrefix">Prefix for zip entries</param>
        private void AddDirectoryToZip(ZipArchive archive, string sourceDir, string entryPrefix)
        {
            foreach (var file in Directory.GetFiles(sourceDir))
            {
                var fileName = Path.GetFileName(file);
                var entryName = string.IsNullOrEmpty(entryPrefix) ? fileName : $"{entryPrefix}/{fileName}";
                
                archive.CreateEntryFromFile(file, entryName);
            }

            foreach (var dir in Directory.GetDirectories(sourceDir))
            {
                var dirName = Path.GetFileName(dir);
                var newPrefix = string.IsNullOrEmpty(entryPrefix) ? dirName : $"{entryPrefix}/{dirName}";
                
                AddDirectoryToZip(archive, dir, newPrefix);
            }
        }
    }
}
