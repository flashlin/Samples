using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterDatabaseSqlTest
{
    [Test]
    public void Alter_database_set_recovery()
    {
        var sql = "ALTER DATABASE ShopDb SET RECOVERY SIMPLE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterDatabaseStatement
        {
            DatabaseName = "ShopDb",
            Setting = "RECOVERY",
            SettingValue = "SIMPLE"
        });
    }

    [Test]
    public void Alter_database_set_single_user()
    {
        var sql = "ALTER DATABASE ShopDb SET SINGLE_USER";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterDatabaseStatement
        {
            DatabaseName = "ShopDb",
            Setting = "SINGLE_USER"
        });
    }
}
