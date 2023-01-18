using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

[WebApiClient("SamApiClient")]
public interface ISamApiClient
{
    [WebApiClientMethod(Method = InvokeMethod.Post, Timeout = "1000")]
    void Test(Request1 req);
}