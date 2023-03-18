using Microsoft.JSInterop;
using System.Threading.Tasks;
using QueryKits.Services;

namespace QueryWeb.Models;
public static class JsInterop
{
    public static ValueTask<object> LoadScript(IJSRuntime jsRuntime, string scriptPath)
    {
        return jsRuntime.InvokeAsync<object>("loadScript", scriptPath);
    }

    public static ValueTask<object> CallTestFunction(IJSRuntime jsRuntime)
    {
        return jsRuntime.InvokeAsync<object>("test");
    }
}

public interface IJsHelper
{
    Task InvokeVoidWithObjectAsync(string method, object obj);
}

public class JsHelper : IJsHelper
{
    private readonly IJSRuntime _jsRuntime;
    private readonly IJsJsonSerializer _jsonSerializer;

    public JsHelper(IJSRuntime jsRuntime, IJsJsonSerializer jsonSerializer)
    {
        _jsonSerializer = jsonSerializer;
        _jsRuntime = jsRuntime;
    }
    
    public async Task InvokeVoidWithObjectAsync(string method, object obj)
    {
        await _jsRuntime.InvokeVoidAsync(method, _jsonSerializer.Serialize(obj));
    }
}
