using T1.SourceGenerator.Attributes;
namespace ConsoleDemoApp;

[AutoMapping(typeof(UserDto), "ToXXX")]
public class UserEntity
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public List<Address> Address1 { get; set; } = new List<Address>();
    public int Level { get; set; }
    public float Price { get; set; }
    public DateTime Birth { get; }
}

public class Request1
{
    public string Id { get; set; }
}

[WebApiClient("SamApiClient")]
public interface ISamApiClient
{
    [WebApiClientMethod(Method = InvokeMethod.Post, Timeout = "1000")]
    void Test(Request1 req);
}