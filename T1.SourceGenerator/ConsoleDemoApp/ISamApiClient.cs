using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

public interface IGlobalSetting
{
    string GetValue(String key);
}

[WebApiClient(ClientClassName = "SamApiClient", Namespace = "ConsoleDemoApp")]
[WebApiClientConstructorInject(typeof(IGlobalSetting), "globalSetting", AssignCode = "globalSetting")]
public interface ISamApiClient
{
    [WebApiClientMethod("mgmt/test", Method = InvokeMethod.Post, Timeout = "1000")]
    void Test(Request1 req);
    
   [WebApiClientMethod("mgmt/test2", Method = InvokeMethod.Get, Timeout = "00:00:30")]
    void Test2();

   [WebApiClientMethod("mgmt/test3")]
   Response1 Test3();
   
   [WebApiClientMethod("mgmt/test4")]
   Response1 Test4(int a);
}

