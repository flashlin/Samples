using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseFulltextAndKeyDdlSqlTest
{
    [Test]
    public void Create_fulltext_catalog_as_default()
    {
        var sql = "CREATE FULLTEXT CATALOG ftCat AS DEFAULT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFulltextCatalogStatement
        {
            Name = "ftCat",
            IsDefault = true
        });
    }

    [Test]
    public void Create_fulltext_stoplist()
    {
        var sql = "CREATE FULLTEXT STOPLIST sl";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateFulltextStoplistStatement
        {
            Name = "sl"
        });
    }

    [Test]
    public void Create_symmetric_key()
    {
        var sql = "CREATE SYMMETRIC KEY sk WITH ALGORITHM = AES_256 ENCRYPTION BY CERTIFICATE c";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSymmetricKeyStatement
        {
            Name = "sk",
            Options = ["ALGORITHM = AES_256", "ENCRYPTION BY CERTIFICATE c"]
        });
    }

    [Test]
    public void Open_symmetric_key()
    {
        var sql = "OPEN SYMMETRIC KEY k DECRYPTION BY CERTIFICATE c";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSymmetricKeyStatement
        {
            IsOpen = true,
            KeyName = "k",
            DecryptionBy = "CERTIFICATE c"
        });
    }

    [Test]
    public void Close_all_symmetric_keys()
    {
        var sql = "CLOSE ALL SYMMETRIC KEYS";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSymmetricKeyStatement
        {
            IsOpen = false,
            AllKeys = true
        });
    }

    [Test]
    public void Alter_fulltext_index_start_full_population()
    {
        var sql = "ALTER FULLTEXT INDEX ON Articles START FULL POPULATION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterFulltextIndexStatement
        {
            TableName = "Articles",
            Action = "START FULL POPULATION"
        });
    }

    [Test]
    public void Alter_server_configuration()
    {
        var sql = "ALTER SERVER CONFIGURATION SET PROCESS AFFINITY CPU = AUTO";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterServerConfigurationStatement
        {
            Setting = "PROCESS AFFINITY CPU = AUTO"
        });
    }
}
