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

        [Test]
        public void GenerateCodeFirstFromSchema_TableWithoutPrimaryKey_HasNoKey()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "AdminIP"
            };
            // Add fields without primary key
            table.Fields.Add(TestHelper.CreateField("IP", "varchar(50)", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("Status", "bit", true, null, false, false));
            table.Fields.Add(TestHelper.CreateField("CreatedOn", "datetime", true, null, false, false));
            table.Fields.Add(TestHelper.CreateField("CreatedBy", "nvarchar(50)", true, null, false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/AdminIPEntityConfiguration.cs"];

            configCode.Should().Contain("builder.HasNoKey();");
            configCode.Should().NotContain("builder.HasKey(");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_TableWithoutPrimaryKey_ConfigurationStillValid()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "LogEntries"
            };
            table.Fields.Add(TestHelper.CreateField("Message", "nvarchar(500)", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("Timestamp", "datetime2", false, null, false, false));
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/LogEntriesEntityConfiguration.cs"];

            // Verify HasNoKey is present
            configCode.Should().Contain("builder.HasNoKey();");
            
            // Verify properties are still configured
            configCode.Should().Contain("builder.Property(x => x.Message)");
            configCode.Should().Contain("builder.Property(x => x.Timestamp)");
            configCode.Should().Contain(".IsRequired()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ComputedColumnPersisted_HasComputedColumnSqlWithStoredTrue()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Orders"
            };
            
            table.Fields.Add(TestHelper.CreateField("OrderId", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Quantity", "int", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("UnitPrice", "decimal(10,2)", false, null, false, false));
            
            // Computed column - PERSISTED
            table.Fields.Add(TestHelper.CreateComputedField("TotalPrice", "decimal(21,2)", true, "([Quantity]*[UnitPrice])", true));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/OrdersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasComputedColumnSql(\"([Quantity]*[UnitPrice])\", stored: true)");
            configCode.Should().NotContain("TotalPrice).HasDefaultValue");
            configCode.Should().NotContain("TotalPrice).IsRequired");
            configCode.Should().NotContain("TotalPrice).HasColumnType");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ComputedColumnNotPersisted_HasComputedColumnSqlWithStoredFalse()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Orders"
            };
            
            table.Fields.Add(TestHelper.CreateField("OrderId", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Quantity", "int", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("UnitPrice", "decimal(10,2)", false, null, false, false));
            
            // Computed column - NOT PERSISTED (virtual)
            table.Fields.Add(TestHelper.CreateComputedField("TotalPriceWithTax", "decimal(21,2)", true, "([Quantity]*[UnitPrice]*(1.1))", false));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/OrdersEntityConfiguration.cs"];

            configCode.Should().Contain(".HasComputedColumnSql(\"([Quantity]*[UnitPrice]*(1.1))\", stored: false)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ComputedColumn_EntityHasProperty()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Orders"
            };
            
            table.Fields.Add(TestHelper.CreateField("OrderId", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateComputedField("TotalPrice", "decimal(21,2)", true, "([Quantity]*[UnitPrice])", true));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var entityCode = result["TestDb/Entities/OrdersEntity.cs"];

            // Computed columns should still have properties in the entity
            entityCode.Should().Contain("public decimal? TotalPrice { get; set; }");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_ComputedColumnWithSpecialCharacters_EscapesQuotes()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Test"
            };
            
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            // SQL expression with quotes (edge case)
            table.Fields.Add(TestHelper.CreateComputedField("ComputedField", "nvarchar(100)", true, "(CASE WHEN [Status] = 'Active' THEN 'Yes' ELSE 'No' END)", true));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/TestEntityConfiguration.cs"];

            // Should escape the quotes properly
            configCode.Should().Contain("HasComputedColumnSql");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_NonIdentityPrimaryKeyWithOtherIdentityField_HasValueGeneratedNever()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "CustomerAccountTier"
            };
            
            // Id is Identity but NOT primary key
            var idField = TestHelper.CreateField("Id", "int", false, null, false, false);
            idField.IsAutoIncrement = true;
            table.Fields.Add(idField);
            
            // CustID is primary key but NOT Identity
            var custIdField = TestHelper.CreateField("CustID", "int", false, null, true, false);
            custIdField.IsPrimaryKey = true;
            custIdField.IsAutoIncrement = false;
            table.Fields.Add(custIdField);
            
            table.Fields.Add(TestHelper.CreateField("UserName", "nvarchar(50)", false, null, false, false));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/CustomerAccountTierEntityConfiguration.cs"];

            // Should have ValueGeneratedNever for non-identity primary key
            configCode.Should().Contain("builder.HasKey(x => x.CustID);");
            configCode.Should().Contain("builder.Property(x => x.CustID)");
            configCode.Should().Contain(".ValueGeneratedNever();");
            
            // Should have ValueGeneratedOnAdd for identity non-primary field
            configCode.Should().Contain("builder.Property(x => x.Id)");
            configCode.Should().Contain(".ValueGeneratedOnAdd()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_IdentityPrimaryKey_NoValueGeneratedNever()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Users"
            };
            
            // Normal case: primary key IS identity
            var idField = TestHelper.CreateField("Id", "int", false, null, true, true);
            idField.IsPrimaryKey = true;
            idField.IsAutoIncrement = true;
            table.Fields.Add(idField);
            
            table.Fields.Add(TestHelper.CreateField("Name", "nvarchar(100)", false, null, false, false));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/UsersEntityConfiguration.cs"];

            // Should NOT have ValueGeneratedNever
            configCode.Should().NotContain(".ValueGeneratedNever()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_CompositePrimaryKeyWithIdentityField_HasValueGeneratedNever()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Orders"
            };
            
            // Identity field (not part of primary key)
            var idField = TestHelper.CreateField("Id", "int", false, null, false, false);
            idField.IsAutoIncrement = true;
            table.Fields.Add(idField);
            
            // Composite primary key (non-identity)
            var orderNoField = TestHelper.CreateField("OrderNo", "int", false, null, false, false);
            orderNoField.IsPrimaryKey = true;
            table.Fields.Add(orderNoField);
            
            var yearField = TestHelper.CreateField("Year", "int", false, null, false, false);
            yearField.IsPrimaryKey = true;
            table.Fields.Add(yearField);
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/OrdersEntityConfiguration.cs"];

            // Composite primary key should both have ValueGeneratedNever
            configCode.Should().Contain("builder.HasKey(x => new { x.OrderNo, x.Year });");
            configCode.Should().Contain("builder.Property(x => x.OrderNo)");
            configCode.Should().Contain(".ValueGeneratedNever();");
            configCode.Should().Contain("builder.Property(x => x.Year)");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_TimestampField_HasIsRowVersion()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Products"
            };
            
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("Name", "nvarchar(100)", false, null, false, false));
            table.Fields.Add(TestHelper.CreateField("TStamp", "timestamp", false, null, false, false));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            
            // Check Entity
            var entityCode = result["TestDb/Entities/ProductsEntity.cs"];
            entityCode.Should().Contain("public required byte[] TStamp { get; set; }");
            
            // Check Configuration
            var configCode = result["TestDb/Configurations/ProductsEntityConfiguration.cs"];
            configCode.Should().Contain("builder.Property(x => x.TStamp)");
            configCode.Should().Contain(".HasColumnType(\"timestamp\")");
            configCode.Should().Contain(".IsRowVersion()");
            configCode.Should().Contain(".ValueGeneratedOnAddOrUpdate()");
            
            // Should NOT have these
            configCode.Should().NotContain("TStamp).IsRequired()");
            configCode.Should().NotContain("TStamp).ValueGeneratedOnAdd()");
        }

        [Test]
        public void GenerateCodeFirstFromSchema_RowversionField_HasIsRowVersion()
        {
            var schema = new T1.EfCodeFirstGenerateCli.Models.DbSchema
            {
                DatabaseName = "TestDb",
                ContextName = "TestDb"
            };
            var table = new T1.EfCodeFirstGenerateCli.Models.TableSchema
            {
                TableName = "Orders"
            };
            
            table.Fields.Add(TestHelper.CreateField("Id", "int", false, null, true, true));
            table.Fields.Add(TestHelper.CreateField("RowVer", "rowversion", false, null, false, false));
            
            schema.Tables.Add(table);

            var result = _generator.GenerateCodeFirstFromSchema(schema, "TestNamespace");
            var configCode = result["TestDb/Configurations/OrdersEntityConfiguration.cs"];
            
            configCode.Should().Contain(".IsRowVersion()");
            configCode.Should().Contain(".ValueGeneratedOnAddOrUpdate()");
        }
    }
}

