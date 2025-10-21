using FluentAssertions;
using NUnit.Framework;
using T1.EfCodeFirstGenerateCli.CodeGenerator;
using T1.EfCodeFirstGenerateCliTest.Helpers;

namespace T1.EfCodeFirstGenerateCliTest.Tests
{
    [TestFixture]
    public class EfCodeGeneratorTests
    {
        private EfCodeGenerator _generator = null!;

        [SetUp]
        public void Setup()
        {
            _generator = new EfCodeGenerator();
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_GeneratesDbContext()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");

            result.Should().ContainKey("TestDb/TestDbDbContext.cs");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_DbContextContainsCorrectNamespace()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var dbContextCode = result["TestDb/TestDbDbContext.cs"];

            dbContextCode.Should().Contain("namespace TestNamespace");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_DbContextContainsDbSets()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var dbContextCode = result["TestDb/TestDbDbContext.cs"];

            dbContextCode.Should().Contain("public DbSet<UsersEntity> Users");
            dbContextCode.Should().Contain("public DbSet<ProductsEntity> Products");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_DbContextAppliesConfigurations()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var dbContextCode = result["TestDb/TestDbDbContext.cs"];

            dbContextCode.Should().Contain("modelBuilder.ApplyConfiguration(new UsersEntityConfiguration())");
            dbContextCode.Should().Contain("modelBuilder.ApplyConfiguration(new ProductsEntityConfiguration())");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_GeneratesEntityClasses()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");

            result.Should().ContainKey("TestDb/Entities/UsersEntity.cs");
            result.Should().ContainKey("TestDb/Entities/ProductsEntity.cs");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_EntityContainsProperties()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var usersEntityCode = result["TestDb/Entities/UsersEntity.cs"];

            usersEntityCode.Should().Contain("public int Id");
            usersEntityCode.Should().Contain("public required string Username");
            usersEntityCode.Should().Contain("public string? Email");
            usersEntityCode.Should().Contain("public DateTime CreatedAt");
            usersEntityCode.Should().Contain("public bool IsActive");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_NonNullableReferenceType_HasRequiredModifier()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var usersEntityCode = result["TestDb/Entities/UsersEntity.cs"];

            usersEntityCode.Should().Contain("public required string Username");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_NullableReferenceType_NoRequiredModifier()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var usersEntityCode = result["TestDb/Entities/UsersEntity.cs"];

            usersEntityCode.Should().Contain("public string? Email");
            usersEntityCode.Should().NotContain("public required string? Email");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_GeneratesEntityConfigurations()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");

            result.Should().ContainKey("TestDb/Configurations/UsersEntityConfiguration.cs");
            result.Should().ContainKey("TestDb/Configurations/ProductsEntityConfiguration.cs");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_ConfigurationImplementsInterface()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain("IEntityTypeConfiguration<UsersEntity>");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_ConfigurationHasTableMapping()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain("builder.ToTable(\"Users\")");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_ConfigurationHasPrimaryKey()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain("builder.HasKey(x => x.Id)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ValidSchema_ConfigurationHasPropertyConfiguration()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain("builder.Property(x => x.Username)");
            configCode.Should().Contain(".IsRequired()");
            configCode.Should().Contain(".HasMaxLength(100)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_AutoIncrementField_ConfigurationHasValueGeneratedOnAdd()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain("builder.Property(x => x.Id)")
                .And.Contain(".ValueGeneratedOnAdd()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_FieldWithSqlFunctionDefault_ConfigurationHasDefaultValueSql()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValueSql(\"getdate()\")");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_FieldWithConstantDefault_ConfigurationHasDefaultValue()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(true)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_NullableField_ConfigurationIsNotRequired()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/ProductsEntityConfiguration.cs"];
            
            // Stock is nullable int
            var stockStart = configCode.IndexOf("builder.Property(x => x.Stock)");
            var nextPropertyStart = configCode.IndexOf("builder.Property(x => x.IsActive)", stockStart);
            var stockPropertyConfig = configCode.Substring(stockStart, nextPropertyStart - stockStart);
            
            stockPropertyConfig.Should().NotContain(".IsRequired()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_CustomNamespace_UsesCorrectNamespace()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "Custom.Namespace");

            result["TestDb/TestDbDbContext.cs"].Should().Contain("namespace Custom.Namespace");
            result["TestDb/Entities/UsersEntity.cs"].Should().Contain("namespace Custom.Namespace");
            result["TestDb/Configurations/UsersEntityConfiguration.cs"].Should().Contain("namespace Custom.Namespace");
        }
    }
}

