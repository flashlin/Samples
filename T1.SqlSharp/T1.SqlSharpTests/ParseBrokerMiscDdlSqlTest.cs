using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseBrokerMiscDdlSqlTest
{
    [Test]
    public void Create_service()
    {
        var sql = "CREATE SERVICE svc ON QUEUE q (ctr)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateServiceStatement
        {
            Name = "svc",
            OnQueue = "q",
            Contracts = ["ctr"]
        });
    }

    [Test]
    public void Create_contract()
    {
        var sql = "CREATE CONTRACT ctr (msg SENT BY ANY)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateContractStatement
        {
            Name = "ctr",
            Messages = ["msg SENT BY ANY"]
        });
    }

    [Test]
    public void Create_message_type()
    {
        var sql = "CREATE MESSAGE TYPE mt VALIDATION = NONE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateMessageTypeStatement
        {
            Name = "mt",
            Validation = "NONE"
        });
    }

    [Test]
    public void Send_on_conversation()
    {
        var sql = "SEND ON CONVERSATION @h MESSAGE TYPE mt (@body)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSendStatement
        {
            ConversationHandle = "@h",
            MessageType = "mt",
            Body = "@body"
        });
    }

    [Test]
    public void Receive_from_queue()
    {
        var sql = "RECEIVE TOP (1) * FROM q";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlReceiveStatement
        {
            Top = "1",
            FromQueue = "q"
        });
    }

    [Test]
    public void Create_endpoint()
    {
        var sql = "CREATE ENDPOINT ep AS TCP (LISTENER_PORT = 5022)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateEndpointStatement
        {
            Name = "ep",
            Protocol = "TCP",
            Options = ["LISTENER_PORT = 5022"]
        });
    }

    [Test]
    public void Create_workload_group()
    {
        var sql = "CREATE WORKLOAD GROUP wg";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateResourceGovernorObjectStatement
        {
            Kind = "WORKLOAD GROUP",
            Name = "wg"
        });
    }

    [Test]
    public void Create_resource_pool()
    {
        var sql = "CREATE RESOURCE POOL rp";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateResourceGovernorObjectStatement
        {
            Kind = "RESOURCE POOL",
            Name = "rp"
        });
    }

    [Test]
    public void Create_column_master_key()
    {
        var sql = "CREATE COLUMN MASTER KEY cmk WITH (KEY_STORE_PROVIDER_NAME = 'x', KEY_PATH = 'y')";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateColumnKeyStatement
        {
            IsMasterKey = true,
            Name = "cmk",
            Options = ["KEY_STORE_PROVIDER_NAME = 'x'", "KEY_PATH = 'y'"]
        });
    }

    [Test]
    public void Readtext_statement()
    {
        var sql = "READTEXT t.col @ptr 0 100";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTextPointerStatement
        {
            Operation = "READTEXT",
            Arguments = ["t.col", "@ptr", "0", "100"]
        });
    }
}
