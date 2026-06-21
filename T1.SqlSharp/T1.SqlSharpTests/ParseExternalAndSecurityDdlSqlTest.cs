using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseExternalAndSecurityDdlSqlTest
{
    [Test]
    public void Create_external_data_source()
    {
        var sql = "CREATE EXTERNAL DATA SOURCE ds WITH (LOCATION = 'hdfs://x')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateExternalStatement
        {
            Kind = "DATA SOURCE",
            Name = "ds",
            Options = ["LOCATION = 'hdfs://x'"]
        });
    }

    [Test]
    public void Create_external_file_format()
    {
        var sql = "CREATE EXTERNAL FILE FORMAT ff WITH (FORMAT_TYPE = DELIMITEDTEXT)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateExternalStatement
        {
            Kind = "FILE FORMAT",
            Name = "ff",
            Options = ["FORMAT_TYPE = DELIMITEDTEXT"]
        });
    }

    [Test]
    public void Create_external_table()
    {
        var sql = "CREATE EXTERNAL TABLE et (id INT) WITH (LOCATION = '/x', DATA_SOURCE = ds)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateExternalStatement
        {
            Kind = "TABLE",
            Name = "et",
            Columns = [new SqlColumnDefinition { ColumnName = "id", DataType = "INT" }],
            Options = ["LOCATION = '/x'", "DATA_SOURCE = ds"]
        });
    }

    [Test]
    public void Create_security_policy()
    {
        var sql = "CREATE SECURITY POLICY sp ADD FILTER PREDICATE dbo.fn(id) ON dbo.t WITH (STATE = ON)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSecurityPolicyStatement
        {
            Name = "sp",
            Predicates = ["FILTER PREDICATE dbo.fn(id) ON dbo.t"],
            Options = ["STATE = ON"]
        });
    }

    [Test]
    public void Create_credential()
    {
        var sql = "CREATE CREDENTIAL cred WITH IDENTITY = 'user', SECRET = 'pw'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateCredentialStatement
        {
            Name = "cred",
            Options = ["IDENTITY = 'user'", "SECRET = 'pw'"]
        });
    }

    [Test]
    public void Create_database_scoped_credential()
    {
        var sql = "CREATE DATABASE SCOPED CREDENTIAL dsc WITH IDENTITY = 'user'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateCredentialStatement
        {
            Name = "dsc",
            IsDatabaseScoped = true,
            Options = ["IDENTITY = 'user'"]
        });
    }
}
