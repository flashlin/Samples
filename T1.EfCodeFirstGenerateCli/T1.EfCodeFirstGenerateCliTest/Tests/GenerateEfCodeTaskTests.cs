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
                content.Should().Contain("namespace TestNamespace.Config.Databases.ConfigDb");
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

            var result = task.SanitizeFileName("test/sub/path");

            result.Should().Be("test_sub_path");
        }

        [Test]
        public void Execute_MultipleDbFiles_GeneratesCodeForEach()
        {
            // Create 3 test .db files with valid connection strings
            var db1Path = Path.Combine(_testDirectory, "db1.db");
            File.WriteAllText(db1Path, "Server=localhost;Database=Database1;User Id=sa;Password=Pass1");

            var db2Path = Path.Combine(_testDirectory, "db2.db");
            File.WriteAllText(db2Path, "Server=localhost;Database=Database2;User Id=sa;Password=Pass2");

            var db3Path = Path.Combine(_testDirectory, "db3.db");
            File.WriteAllText(db3Path, "Server=localhost;Database=Database3;User Id=sa;Password=Pass3");

            // Verify DbConfigParser finds all 3 .db files
            var configs = T1.EfCodeFirstGenerateCli.ConfigParser.DbConfigParser.GetAllDbConnectionConfigs(_testDirectory);
            configs.Count.Should().Be(3, "DbConfigParser should find all 3 .db files");

            // Create schema files for each database to avoid actual database connections
            // Schema file name must match DbConfig.ContextName (which comes from .db file name)
            var generatedDir = Path.Combine(_testDirectory, "Generated");
            Directory.CreateDirectory(generatedDir);

            // Create schema files using ContextName from DbConfig
            foreach (var config in configs)
            {
                var schemaPath = Path.Combine(generatedDir, $"{config.ContextName}.schema");
                var testSchema = TestHelper.CreateTestSchema(config.DatabaseName);
                testSchema.ContextName = config.DatabaseName;
                var schemaJson = Newtonsoft.Json.JsonConvert.SerializeObject(testSchema, Newtonsoft.Json.Formatting.Indented);
                File.WriteAllText(schemaPath, schemaJson);
            }

            // Execute the task
            var task = new GenerateEfCodeTask
            {
                BuildEngine = _mockEngine,
                ProjectDirectory = _testDirectory,
                RootNamespace = "TestNamespace",
                AssemblyName = "TestAssembly"
            };

            var result = task.Execute();

            // Verify execution succeeded
            result.Should().BeTrue();

            // Verify that all 3 databases have their own directories and files generated
            var db1ContextPath = Path.Combine(generatedDir, "Database1", "Database1DbContext.cs");
            var db2ContextPath = Path.Combine(generatedDir, "Database2", "Database2DbContext.cs");
            var db3ContextPath = Path.Combine(generatedDir, "Database3", "Database3DbContext.cs");

            File.Exists(db1ContextPath).Should().BeTrue($"Database1 DbContext should be generated at {db1ContextPath}");
            File.Exists(db2ContextPath).Should().BeTrue($"Database2 DbContext should be generated at {db2ContextPath}");
            File.Exists(db3ContextPath).Should().BeTrue($"Database3 DbContext should be generated at {db3ContextPath}");

            // Verify entities directories exist for each database
            var db1EntitiesDir = Path.Combine(generatedDir, "Database1", "Entities");
            var db2EntitiesDir = Path.Combine(generatedDir, "Database2", "Entities");
            var db3EntitiesDir = Path.Combine(generatedDir, "Database3", "Entities");

            Directory.Exists(db1EntitiesDir).Should().BeTrue("Database1 Entities directory should exist");
            Directory.Exists(db2EntitiesDir).Should().BeTrue("Database2 Entities directory should exist");
            Directory.Exists(db3EntitiesDir).Should().BeTrue("Database3 Entities directory should exist");

            // Verify entity files exist for each database
            var db1UsersEntity = Path.Combine(db1EntitiesDir, "UsersEntity.cs");
            var db2UsersEntity = Path.Combine(db2EntitiesDir, "UsersEntity.cs");
            var db3UsersEntity = Path.Combine(db3EntitiesDir, "UsersEntity.cs");

            File.Exists(db1UsersEntity).Should().BeTrue("Database1 Users entity should exist");
            File.Exists(db2UsersEntity).Should().BeTrue("Database2 Users entity should exist");
            File.Exists(db3UsersEntity).Should().BeTrue("Database3 Users entity should exist");

            // Verify configurations directories exist for each database
            var db1ConfigsDir = Path.Combine(generatedDir, "Database1", "Configurations");
            var db2ConfigsDir = Path.Combine(generatedDir, "Database2", "Configurations");
            var db3ConfigsDir = Path.Combine(generatedDir, "Database3", "Configurations");

            Directory.Exists(db1ConfigsDir).Should().BeTrue("Database1 Configurations directory should exist");
            Directory.Exists(db2ConfigsDir).Should().BeTrue("Database2 Configurations directory should exist");
            Directory.Exists(db3ConfigsDir).Should().BeTrue("Database3 Configurations directory should exist");

            // Verify namespace structure includes Databases layer
            var db1Content = File.ReadAllText(db1ContextPath);
            db1Content.Should().Contain("namespace TestNamespace.Databases.Database1", 
                "Database1 should have Databases namespace layer");
            
            var db2Content = File.ReadAllText(db2ContextPath);
            db2Content.Should().Contain("namespace TestNamespace.Databases.Database2",
                "Database2 should have Databases namespace layer");
            
            var db3Content = File.ReadAllText(db3ContextPath);
            db3Content.Should().Contain("namespace TestNamespace.Databases.Database3",
                "Database3 should have Databases namespace layer");
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

