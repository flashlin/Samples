using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateTypeSqlTest
{
    [Test]
    public void Create_type_from_base()
    {
        var sql = "CREATE TYPE Phone FROM VARCHAR(20) NOT NULL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTypeStatement
        {
            TypeName = "Phone",
            BaseType = "VARCHAR",
            BaseSize = new SqlDataSize { Size = "20" }
        });
    }

    [Test]
    public void Create_type_as_table()
    {
        var sql = "CREATE TYPE OrderTable AS TABLE (Id INT, Qty INT)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateTypeStatement
        {
            TypeName = "OrderTable",
            IsTable = true,
            TableColumns =
            [
                new SqlColumnDefinition { ColumnName = "Id", DataType = "INT" },
                new SqlColumnDefinition { ColumnName = "Qty", DataType = "INT" }
            ]
        });
    }
}
