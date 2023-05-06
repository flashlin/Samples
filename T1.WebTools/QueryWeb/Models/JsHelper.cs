using Microsoft.JSInterop;
using QueryKits.Services;

namespace QueryWeb.Models;

public class JsHelper : IJsHelper
{
    private readonly IJSRuntime _jsRuntime;
    private readonly IJsJsonSerializer _jsonSerializer;

    public JsHelper(IJSRuntime jsRuntime, IJsJsonSerializer jsonSerializer)
    {
        _jsonSerializer = jsonSerializer;
        _jsRuntime = jsRuntime;
    }
    
    public async Task InvokeVoidWithObjectAsync(string method, object obj, object? obj1)
    {
        //await _jsRuntime.InvokeVoidAsync(method, _jsonSerializer.Serialize(obj), obj1);
        await _jsRuntime.InvokeVoidAsync(method, obj, obj1);
    }
    
    public async Task<bool> ShowMessageAsync(string message)
    {
        var result = await _jsRuntime.InvokeAsync<bool>("messageBox.show", message);
        return result;
    }
}