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

        [Test]
        public void GenerateCodeFirstFromSchema_DbContext_HasConstructorWithOptions()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var dbContextCode = result["TestDb/TestDbDbContext.cs"];

            dbContextCode.Should().Contain("public TestDbDbContext(DbContextOptions<TestDbDbContext> options)");
            dbContextCode.Should().Contain(": base(options)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_DbContext_HasParameterlessConstructor()
        {
            var schema = TestHelper.CreateTestSchema("TestDb");

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var dbContextCode = result["TestDb/TestDbDbContext.cs"];

            dbContextCode.Should().Contain("public TestDbDbContext()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_LowercaseFieldName_ConvertsToPascalCase()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "testTable"
            };
            table.Fields.Add(TestHelper.CreateField("userid", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("username", "nvarchar(100)", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("created_at", "datetime2", false, null, false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var entityCode = result["TestDb/Entities/testTableEntity.cs"];
            var configCode = result["TestDb/Configurations/testTableEntityConfiguration.cs"];

            entityCode.Should().Contain("public int Userid { get; set; }");
            entityCode.Should().Contain("public required string Username { get; set; }");
            entityCode.Should().Contain("public DateTime Created_at { get; set; }");
            
            configCode.Should().Contain("builder.HasKey(x => x.Userid);");
            configCode.Should().Contain("builder.Property(x => x.Userid)");
            configCode.Should().Contain(".HasColumnName(\"userid\")");
            configCode.Should().Contain("builder.Property(x => x.Username)");
            configCode.Should().Contain(".HasColumnName(\"username\")");
            configCode.Should().Contain("builder.Property(x => x.Created_at)");
            configCode.Should().Contain(".HasColumnName(\"created_at\")");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_CompositePrimaryKey_ConvertsToPascalCase()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "orderLines"
            };
            table.Fields.Add(TestHelper.CreateField("orderid", "int", false, null, true, false));
            table.Fields.Add(TestHelper.CreateField("lineid", "int", false, null, true, false));
            table.Fields.Add(TestHelper.CreateField("product_name", "nvarchar(200)", false, null, false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/orderLinesEntityConfiguration.cs"];

            configCode.Should().Contain("builder.HasKey(x => new { x.Orderid, x.Lineid });");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_DecimalFieldWithDefaultValue_HasDecimalSuffix()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Currencies"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("CasinoPlayableLimit", "decimal(19,2)", true, "0", false, false));
            table.Fields.Add(TestHelper.CreateField("Rate", "decimal(18,4)", false, "1.5", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/CurrenciesEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(0m)");
            configCode.Should().Contain(".HasDefaultValue(1.5m)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_FloatFieldWithDefaultValue_HasFloatSuffix()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Measurements"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Temperature", "float", true, "0", false, false));
            table.Fields.Add(TestHelper.CreateField("Pressure", "real", false, "1.5", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/MeasurementsEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(0f)");
            configCode.Should().Contain(".HasDefaultValue(1.5f)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_BigIntFieldWithDefaultValue_HasLongSuffix()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "LargeNumbers"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Counter", "bigint", true, "0", false, false));
            table.Fields.Add(TestHelper.CreateField("MaxValue", "bigint", false, "9999999999", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/LargeNumbersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(0L)");
            configCode.Should().Contain(".HasDefaultValue(9999999999L)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_IntFieldWithDefaultValue_NoSuffix()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Counters"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Value", "int", false, "100", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/CountersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(100)");
            configCode.Should().NotContain("100L");
            configCode.Should().NotContain("100m");
            configCode.Should().NotContain("100f");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_NumericFieldWithDefaultValue_HasDecimalSuffix()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Prices"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Amount", "numeric(10,2)", false, "0", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/PricesEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue(0m)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_TinyIntFieldWithDefaultValue_HasByteCast()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "TransInfo"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("BankFeeStatus", "tinyint", true, "1", false, false));
            table.Fields.Add(TestHelper.CreateField("Status", "tinyint", false, "0", false, false));
            table.Fields.Add(TestHelper.CreateField("MaxValue", "tinyint", false, "255", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/TransInfoEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue((byte)1)");
            configCode.Should().Contain(".HasDefaultValue((byte)0)");
            configCode.Should().Contain(".HasDefaultValue((byte)255)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_SmallIntFieldWithDefaultValue_HasShortCast()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Settings"
            };
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Timeout", "smallint", true, "10", false, false));
            table.Fields.Add(TestHelper.CreateField("RetryCount", "smallint", false, "0", false, false));
            table.Fields.Add(TestHelper.CreateField("MaxConnections", "smallint", false, "100", false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/SettingsEntityConfiguration.cs"];

            configCode.Should().Contain(".HasDefaultValue((short)10)");
            configCode.Should().Contain(".HasDefaultValue((short)0)");
            configCode.Should().Contain(".HasDefaultValue((short)100)");
        }
    }
}

