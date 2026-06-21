using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterDatabaseSqlTest
{
    [Test]
    public void Alter_database_add_file()
    {
        var sql = "ALTER DATABASE ShopDb ADD FILE (NAME = dat2, FILENAME = 'D:\\dat2.ndf')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterDatabaseStatement
        {
            DatabaseName = "ShopDb",
            FileAction = "ADD FILE",
            FileSpec = "(NAME = dat2, FILENAME = 'D:\\dat2.ndf')"
        });
    }

    [Test]
    public void Alter_database_modify_file()
    {
        var sql = "ALTER DATABASE ShopDb MODIFY FILE (NAME = dat2, SIZE = 100)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterDatabaseStatement
        {
            DatabaseName = "ShopDb",
            FileAction = "MODIFY FILE",
            FileSpec = "(NAME = dat2, SIZE = 100)"
        });
    }

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
