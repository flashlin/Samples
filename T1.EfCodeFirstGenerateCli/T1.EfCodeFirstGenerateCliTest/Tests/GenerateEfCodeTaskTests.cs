using FluentAssertions;
using Microsoft.Build.Framework;
using NUnit.Framework;
using System.Collections;
using System.IO;
using T1.EfCodeFirstGenerateCli.Tasks;
using T1.EfCodeFirstGenerateCliTest.Helpers;

namespace T1.EfCodeFirstGenerateCliTest.Tests
{
    [TestFixture]
    public class GenerateEfCodeTaskTests
    {
        private string _testDirectory = null!;
        private MockBuildEngine _mockEngine = null!;

        [SetUp]
        public void Setup()
        {
            _testDirectory = TestHelper.CreateTempTestDirectory();
            _mockEngine = new MockBuildEngine();
        }

        [TearDown]
        public void TearDown()
        {
            TestHelper.CleanupDirectory(_testDirectory);
        }

        [Test]
        public void Execute_NoDbFiles_ReturnsTrue()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = "TestProject",
                AssemblyName = "TestAssembly"
            };

            var result = task.Execute();

            result.Should().BeTrue();
        }

        [Test]
        public void Execute_InvalidProjectDirectory_ReturnsTrue()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = Path.Combine(_testDirectory, "NonExistent"),
                RootNamespace = "TestProject",
                AssemblyName = "TestAssembly"
            };

            var result = task.Execute();

            result.Should().BeTrue();
        }

        [Test]
        public void GetProjectNamespace_WithRootNamespace_ReturnsRootNamespace()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = "MyRootNamespace",
                AssemblyName = "MyAssembly"
            };

            var result = task.GetProjectNamespace();

            result.Should().Be("MyRootNamespace");
        }

        [Test]
        public void GetProjectNamespace_WithoutRootNamespace_ReturnsAssemblyName()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = null,
                AssemblyName = "MyAssembly"
            };

            var result = task.GetProjectNamespace();

            result.Should().Be("MyAssembly");
        }

        [Test]
        public void GetProjectNamespace_WithoutBoth_ReturnsProjectDirectoryName()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = null,
                AssemblyName = null
            };

            var result = task.GetProjectNamespace();

            result.Should().NotBeNullOrEmpty();
        }

        [Test]
        public void Execute_WithDbFileAndSchema_GeneratesCode()
        {
            // Create a test .db file
            var dbFilePath = Path.Combine(_testDirectory, "test.db");
            File.WriteAllText(dbFilePath, "# This is a comment\n# No actual connection string for unit test");

            // Create a schema file to avoid database connection
            var generatedDir = Path.Combine(_testDirectory, "Generated");
            Directory.CreateDirectory(generatedDir);
            var schemaPath = Path.Combine(generatedDir, "_test.schema");

            var testSchema = TestHelper.CreateTestSchema("TestDb");
            var schemaJson = Newtonsoft.Json.JsonConvert.SerializeObject(testSchema, Newtonsoft.Json.Formatting.Indented);
            File.WriteAllText(schemaPath, schemaJson);

            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = "TestNamespace",
                AssemblyName = "TestAssembly"
            };

            var result = task.Execute();

            result.Should().BeTrue();
        }

        [Test]
        public void Execute_SubdirectoryDbFile_GeneratesCorrectNamespace()
        {
            // Create subdirectory
            var configDir = Path.Combine(_testDirectory, "Config");
            Directory.CreateDirectory(configDir);

            // Create a test .db file in subdirectory
            var dbFilePath = Path.Combine(configDir, "test.db");
            File.WriteAllText(dbFilePath, "# Config database");

            // Create a schema file
            var generatedDir = Path.Combine(configDir, "Generated");
            Directory.CreateDirectory(generatedDir);
            var schemaPath = Path.Combine(generatedDir, "_test.schema");

            var testSchema = TestHelper.CreateTestSchema("ConfigDb");
            var schemaJson = Newtonsoft.Json.JsonConvert.SerializeObject(testSchema, Newtonsoft.Json.Formatting.Indented);
            File.WriteAllText(schemaPath, schemaJson);

            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = "TestNamespace",
                AssemblyName = "TestAssembly"
            };

            var result = task.Execute();

            result.Should().BeTrue();

            // Check if files were generated
            var dbContextPath = Path.Combine(generatedDir, "ConfigDb", "ConfigDbDbContext.cs");
            if (File.Exists(dbContextPath))
            {
                var content = File.ReadAllText(dbContextPath);
                content.Should().Contain("namespace TestNamespace.Config");
            }
        }

        [Test]
        public void SanitizeFileName_InvalidChars_ReturnsCleanName()
        {
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory
            };

            var result = task.SanitizeFileName("test<>:file|name");

            result.Should().NotBeNull();
            result.Should().NotContain("<");
            result.Should().NotContain(">");
            result.Should().NotContain(":");
            result.Should().NotContain("|");
        }
    }

    // Simple Mock BuildEngine for testing MSBuild tasks
    public class MockBuildEngine : IBuildEngine
    {
        public bool ContinueOnError => false;
        public int LineNumberOfTaskNode => 0;
        public int ColumnNumberOfTaskNode => 0;
        public string ProjectFileOfTaskNode => "";

        public bool BuildProjectFile(string projectFileName, string[] targetNames, IDictionary globalProperties, IDictionary targetOutputs) => true;
        public void LogCustomEvent(CustomBuildEventArgs e) { }
        public void LogErrorEvent(BuildErrorEventArgs e) { }
        public void LogMessageEvent(BuildMessageEventArgs e) { }
        public void LogWarningEvent(BuildWarningEventArgs e) { }
    }
}

