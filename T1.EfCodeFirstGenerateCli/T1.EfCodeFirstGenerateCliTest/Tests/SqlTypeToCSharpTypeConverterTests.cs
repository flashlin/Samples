using FluentAssertions;
using NUnit.Framework;
using T1.EfCodeFirstGenerateCli.Converters;

namespace T1.EfCodeFirstGenerateCliTest.Tests
{
    [TestFixture]
    public class SqlTypeToCSharpTypeConverterTests
    {
        private SqlTypeToCSharpTypeConverter _converter = null!;

        [SetUp]
        public void Setup()
        {
            _converter = new SqlTypeToCSharpTypeConverter();
        }

        [Test]
        public void Convert_Int_ReturnsInt()
        {
            var result = _converter.ConvertType("int", false);
            result.Should().Be("int");
        }

        [Test]
        public void Convert_NullableInt_ReturnsNullableInt()
        {
            var result = _converter.ConvertType("int", true);
            result.Should().Be("int?");
        }

        [Test]
        public void Convert_BigInt_ReturnsLong()
        {
            var result = _converter.ConvertType("bigint", false);
            result.Should().Be("long");
        }

        [Test]
        public void Convert_NullableBigInt_ReturnsNullableLong()
        {
            var result = _converter.ConvertType("bigint", true);
            result.Should().Be("long?");
        }

        [Test]
        public void Convert_Varchar_ReturnsString()
        {
            var result = _converter.ConvertType("varchar(100)", false);
            result.Should().Be("string");
        }

        [Test]
        public void Convert_NullableVarchar_ReturnsString()
        {
            // Note: Reference types like string are handled separately
            // by the generator, not by the converter
            var result = _converter.ConvertType("varchar(100)", true);
            result.Should().Be("string?");
        }

        [Test]
        public void Convert_Nvarchar_ReturnsString()
        {
            var result = _converter.ConvertType("nvarchar(255)", false);
            result.Should().Be("string");
        }

        [Test]
        public void Convert_Text_ReturnsString()
        {
            var result = _converter.ConvertType("text", false);
            result.Should().Be("string");
        }

        [Test]
        public void Convert_Decimal_ReturnsDecimal()
        {
            var result = _converter.ConvertType("decimal(18,2)", false);
            result.Should().Be("decimal");
        }

        [Test]
        public void Convert_NullableDecimal_ReturnsNullableDecimal()
        {
            var result = _converter.ConvertType("decimal(18,2)", true);
            result.Should().Be("decimal?");
        }

        [Test]
        public void Convert_Bit_ReturnsBool()
        {
            var result = _converter.ConvertType("bit", false);
            result.Should().Be("bool");
        }

        [Test]
        public void Convert_NullableBit_ReturnsNullableBool()
        {
            var result = _converter.ConvertType("bit", true);
            result.Should().Be("bool?");
        }

        [Test]
        public void Convert_DateTime_ReturnsDateTime()
        {
            var result = _converter.ConvertType("datetime", false);
            result.Should().Be("DateTime");
        }

        [Test]
        public void Convert_DateTime2_ReturnsDateTime()
        {
            var result = _converter.ConvertType("datetime2", false);
            result.Should().Be("DateTime");
        }

        [Test]
        public void Convert_NullableDateTime_ReturnsNullableDateTime()
        {
            var result = _converter.ConvertType("datetime", true);
            result.Should().Be("DateTime?");
        }

        [Test]
        public void Convert_UniqueIdentifier_ReturnsGuid()
        {
            var result = _converter.ConvertType("uniqueidentifier", false);
            result.Should().Be("Guid");
        }

        [Test]
        public void Convert_NullableUniqueIdentifier_ReturnsNullableGuid()
        {
            var result = _converter.ConvertType("uniqueidentifier", true);
            result.Should().Be("Guid?");
        }

        [Test]
        public void Convert_Binary_ReturnsByteArray()
        {
            var result = _converter.ConvertType("binary(16)", false);
            result.Should().Be("byte[]");
        }

        [Test]
        public void Convert_Varbinary_ReturnsByteArray()
        {
            var result = _converter.ConvertType("varbinary(max)", false);
            result.Should().Be("byte[]");
        }

        [Test]
        public void Convert_NullableBinary_ReturnsByteArray()
        {
            // Note: byte[] is a reference type, handled separately
            var result = _converter.ConvertType("binary(16)", true);
            result.Should().Be("byte[]");
        }

        [Test]
        public void Convert_SmallInt_ReturnsShort()
        {
            var result = _converter.ConvertType("smallint", false);
            result.Should().Be("short");
        }

        [Test]
        public void Convert_TinyInt_ReturnsByte()
        {
            var result = _converter.ConvertType("tinyint", false);
            result.Should().Be("byte");
        }

        [Test]
        public void Convert_Float_ReturnsDouble()
        {
            var result = _converter.ConvertType("float", false);
            result.Should().Be("double");
        }

        [Test]
        public void Convert_Real_ReturnsDouble()
        {
            // Note: Both float and real map to double in this implementation
            var result = _converter.ConvertType("real", false);
            result.Should().Be("double");
        }

        [Test]
        public void RegisterCustomMapping_CustomType_ReturnsCustomCSharpType()
        {
            _converter.RegisterCustomMapping("geometry", (sqlType, isNullable) =>
                isNullable ? "CustomGeometry?" : "CustomGeometry");

            var result = _converter.ConvertType("geometry", false);
            result.Should().Be("CustomGeometry");
        }

        [Test]
        public void RegisterCustomMapping_NullableCustomType_ReturnsNullableCustomCSharpType()
        {
            _converter.RegisterCustomMapping("geometry", (sqlType, isNullable) =>
                isNullable ? "CustomGeometry?" : "CustomGeometry");

            var result = _converter.ConvertType("geometry", true);
            result.Should().Be("CustomGeometry?");
        }
    }
}

