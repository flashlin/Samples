using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDeclareSqlTest
{
    [Test]
    public void Declare_single_variable()
    {
        var sql = "DECLARE @count INT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration { Name = "@count", DataType = "INT" }
            ]
        });
    }

    [Test]
    public void Declare_with_initial_value()
    {
        var sql = "DECLARE @count INT = 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration
                {
                    Name = "@count",
                    DataType = "INT",
                    InitialValue = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                }
            ]
        });
    }

    [Test]
    public void Declare_multiple_variables_with_size()
    {
        var sql = "DECLARE @a INT, @name VARCHAR(50) = 'x'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDeclareStatement
        {
            Declarations =
            [
                new SqlVariableDeclaration { Name = "@a", DataType = "INT" },
                new SqlVariableDeclaration
                {
                    Name = "@name",
                    DataType = "VARCHAR",
                    DataSize = new SqlDataSize { Size = "50" },
                    InitialValue = new SqlValue { SqlType = SqlType.String, Value = "'x'" }
                }
            ]
        });
    }
}
