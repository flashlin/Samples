using FluentAssertions;
using NUnit.Framework;
using System.Collections.Generic;
using System.IO;
using T1.EfCodeFirstGenerateCli.Common;

namespace T1.EfCodeFirstGenerateCliTest.Tests
{
    [TestFixture]
    public class FileWriterHelperTests
    {
        private string _testDir = null!;
        
        [SetUp]
        public void Setup()
        {
            _testDir = Path.Combine(Path.GetTempPath(), "FileWriterHelperTests_" + Path.GetRandomFileName());
            Directory.CreateDirectory(_testDir);
        }
        
        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_testDir))
            {
                Directory.Delete(_testDir, true);
            }
        }
        
        [Test]
        public void WriteGeneratedFiles_NewFiles_WritesAllFiles()
        {
            var files = new Dictionary<string, string>
            {
                ["TestDb/TestFile1.cs"] = "test content 1",
                ["TestDb/TestFile2.cs"] = "test content 2"
            };
            
            var count = FileWriterHelper.WriteGeneratedFiles(files, _testDir);
            
            count.Should().Be(2);
            File.Exists(Path.Combine(_testDir, "TestDb/TestFile1.cs")).Should().BeTrue();
            File.Exists(Path.Combine(_testDir, "TestDb/TestFile2.cs")).Should().BeTrue();
        }
        
        [Test]
        public void WriteGeneratedFiles_ExistingFile_SkipsFile()
        {
            var testFile = Path.Combine(_testDir, "TestDb/TestFile.cs");
            Directory.CreateDirectory(Path.GetDirectoryName(testFile)!);
            File.WriteAllText(testFile, "old content");
            
            var files = new Dictionary<string, string>
            {
                ["TestDb/TestFile.cs"] = "new content"
            };
            
            var count = FileWriterHelper.WriteGeneratedFiles(files, _testDir);
            
            count.Should().Be(0); // Should skip
            File.ReadAllText(testFile).Should().Be("old content"); // Original content preserved
        }
        
        [Test]
        public void WriteGeneratedFiles_MixedNewAndExisting_WritesOnlyNew()
        {
            var existingFile = Path.Combine(_testDir, "TestDb/ExistingFile.cs");
            Directory.CreateDirectory(Path.GetDirectoryName(existingFile)!);
            File.WriteAllText(existingFile, "existing content");
            
            var files = new Dictionary<string, string>
            {
                ["TestDb/ExistingFile.cs"] = "new content",
                ["TestDb/NewFile.cs"] = "new file content"
            };
            
            var count = FileWriterHelper.WriteGeneratedFiles(files, _testDir);
            
            count.Should().Be(1); // Only new file written
            File.ReadAllText(existingFile).Should().Be("existing content"); // Existing file unchanged
            File.ReadAllText(Path.Combine(_testDir, "TestDb/NewFile.cs")).Should().Be("new file content");
        }
        
        [Test]
        public void WriteGeneratedFiles_CreatesDirectoryIfNotExists()
        {
            var files = new Dictionary<string, string>
            {
                ["TestDb/SubDir1/SubDir2/TestFile.cs"] = "content"
            };
            
            var count = FileWriterHelper.WriteGeneratedFiles(files, _testDir);
            
            count.Should().Be(1);
            File.Exists(Path.Combine(_testDir, "TestDb/SubDir1/SubDir2/TestFile.cs")).Should().BeTrue();
        }
    }
}

