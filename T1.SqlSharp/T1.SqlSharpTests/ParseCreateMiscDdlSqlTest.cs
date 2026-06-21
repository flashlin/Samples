using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateMiscDdlSqlTest
{
    [Test]
    public void Create_fulltext_index()
    {
        var sql = "CREATE FULLTEXT INDEX ON Articles (Title, Body) KEY INDEX PK_Articles ON ftCatalog";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFulltextIndexStatement
        {
            TableName = "Articles",
            Columns = ["Title", "Body"],
            KeyIndex = "PK_Articles",
            Catalog = "ftCatalog"
        });
    }

    [Test]
    public void Create_partition_function_range_left()
    {
        var sql = "CREATE PARTITION FUNCTION RangePF (int) AS RANGE LEFT FOR VALUES (1, 100, 1000)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePartitionFunctionStatement
        {
            FunctionName = "RangePF",
            InputType = "int",
            RangeDirection = "LEFT",
            BoundaryValues = ["1", "100", "1000"]
        });
    }

    [Test]
    public void Create_partition_scheme()
    {
        var sql = "CREATE PARTITION SCHEME RangePS AS PARTITION RangePF TO (fg1, fg2, fg3)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePartitionSchemeStatement
        {
            SchemeName = "RangePS",
            PartitionFunction = "RangePF",
            FileGroups = ["fg1", "fg2", "fg3"]
        });
    }

    [Test]
    public void Create_partition_scheme_all()
    {
        var sql = "CREATE PARTITION SCHEME RangePS AS PARTITION RangePF ALL TO (fg1)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePartitionSchemeStatement
        {
            SchemeName = "RangePS",
            PartitionFunction = "RangePF",
            AllToOneFileGroup = true,
            FileGroups = ["fg1"]
        });
    }

    [Test]
    public void Create_xml_schema_collection()
    {
        var sql = "CREATE XML SCHEMA COLLECTION MySchema AS '<xsd:schema/>'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateXmlSchemaCollectionStatement
        {
            Name = "MySchema",
            Schema = "'<xsd:schema/>'"
        });
    }
}
