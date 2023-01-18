using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

public interface IGlobalSetting
{
    string GetValue(String key);
}

[WebApiClient(ClientClassName = "SamApiClient", Namespace = "ConsoleDemoApp")]
//[AutoConstructorInject(typeof(IGlobalSetting), "globalSetting", "globalSetting")]
public interface ISamApiClient
{
    [WebApiClientMethod("mgmt/test", Method = InvokeMethod.Post, Timeout = "1000")]
    void Test(Request1 req);
    
   [WebApiClientMethod("mgmt/test2", Timeout = "00:00:30")]
    void Test2();

   [WebApiClientMethod("mgmt/test3")]
   Response1 Test3();
   
   [WebApiClientMethod("mgmt/test4")]
   Response1 Test4(int a);
}

