namespace T1.SourceGenerator.Attributes;

[AttributeUsage(AttributeTargets.Interface, Inherited = false, AllowMultiple = false)]
public class WebApiClientAttribute : Attribute
{
    public WebApiClientAttribute(string clientClassName)
    {
        ClientClassName = clientClassName;
    }   
    
    public string ClientClassName { get; set; }
}

public enum InvokeMethod
{
    Post,
    Get
}

[AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)]
public class WebApiClientMethodAttribute : Attribute
{
    public WebApiClientMethodAttribute(string apiPath)
    {
        ApiPath = apiPath;
    }
    public string ApiPath { get; set; }
    public InvokeMethod Method { get; set; } = InvokeMethod.Post;
    public string Timeout { get; set; } = "00:00:30";
}