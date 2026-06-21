using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterSchemaSqlTest
{
    [Test]
    public void Alter_schema_transfer()
    {
        var sql = "ALTER SCHEMA Sales TRANSFER dbo.Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterSchemaStatement
        {
            SchemaName = "Sales",
            ObjectName = "dbo.Orders"
        });
    }
}
