using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseSecurityDdlSqlTest
{
    [Test]
    public void Create_certificate_from_file()
    {
        var sql = "CREATE CERTIFICATE MyCert FROM FILE = 'C:\\cert.cer'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateCertificateStatement
        {
            Name = "MyCert",
            FromFile = "'C:\\cert.cer'"
        });
    }

    [Test]
    public void Create_certificate_with_subject_and_password()
    {
        var sql = "CREATE CERTIFICATE MyCert ENCRYPTION BY PASSWORD = 'pw' WITH SUBJECT = 'My Cert'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateCertificateStatement
        {
            Name = "MyCert",
            Password = "'pw'",
            Options = ["SUBJECT = 'My Cert'"]
        });
    }

    [Test]
    public void Create_master_key()
    {
        var sql = "CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'StrongPwd'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateMasterKeyStatement
        {
            Password = "'StrongPwd'"
        });
    }

    [Test]
    public void Setuser_with_name()
    {
        var sql = "SETUSER 'guest'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetUserStatement
        {
            UserName = "'guest'"
        });
    }

    [Test]
    public void Setuser_revert()
    {
        var sql = "SETUSER";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetUserStatement());
    }
}
