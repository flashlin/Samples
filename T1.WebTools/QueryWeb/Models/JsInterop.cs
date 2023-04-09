using Microsoft.JSInterop;
using System.Threading.Tasks;

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