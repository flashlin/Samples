using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseConversationAndCreateObjectSqlTest
{
    [Test]
    public void Begin_dialog_conversation()
    {
        "BEGIN DIALOG CONVERSATION @h FROM SERVICE s1 TO SERVICE 's2'".ParseSql().ShouldBe(new SqlConversationStatement
        {
            Operation = "BEGIN DIALOG CONVERSATION",
            Handle = "@h",
            Action = "FROM SERVICE s1 TO SERVICE 's2'"
        });
    }

    [Test]
    public void End_conversation()
    {
        "END CONVERSATION @h".ParseSql().ShouldBe(new SqlConversationStatement
        {
            Operation = "END CONVERSATION",
            Handle = "@h"
        });
    }

    [Test]
    public void Move_conversation()
    {
        "MOVE CONVERSATION @h TO @group".ParseSql().ShouldBe(new SqlConversationStatement
        {
            Operation = "MOVE CONVERSATION",
            Handle = "@h",
            Action = "TO @group"
        });
    }

    [Test]
    public void Get_conversation_group()
    {
        "GET CONVERSATION GROUP @g FROM q".ParseSql().ShouldBe(new SqlConversationStatement
        {
            Operation = "GET CONVERSATION GROUP",
            Handle = "@g",
            Action = "FROM q"
        });
    }

    [Test]
    public void Create_event_session()
    {
        "CREATE EVENT SESSION s ON SERVER".ParseSql().ShouldBe(new SqlCreateObjectStatement
        {
            Kind = "EVENT SESSION", Name = "s", Action = "ON SERVER"
        });
    }

    [Test]
    public void Create_route()
    {
        "CREATE ROUTE r".ParseSql().ShouldBe(new SqlCreateObjectStatement { Kind = "ROUTE", Name = "r" });
    }

    [Test]
    public void Create_remote_service_binding()
    {
        "CREATE REMOTE SERVICE BINDING b".ParseSql().ShouldBe(new SqlCreateObjectStatement
        {
            Kind = "REMOTE SERVICE BINDING", Name = "b"
        });
    }

    [Test]
    public void Create_event_notification()
    {
        "CREATE EVENT NOTIFICATION n".ParseSql().ShouldBe(new SqlCreateObjectStatement
        {
            Kind = "EVENT NOTIFICATION", Name = "n"
        });
    }

    [Test]
    public void Create_application_role()
    {
        "CREATE APPLICATION ROLE ar WITH PASSWORD = 'x'".ParseSql().ShouldBe(new SqlCreateObjectStatement
        {
            Kind = "APPLICATION ROLE", Name = "ar", Action = "WITH PASSWORD = 'x'"
        });
    }

    [Test]
    public void Create_server_role()
    {
        "CREATE SERVER ROLE sr".ParseSql().ShouldBe(new SqlCreateObjectStatement { Kind = "SERVER ROLE", Name = "sr" });
    }
}
