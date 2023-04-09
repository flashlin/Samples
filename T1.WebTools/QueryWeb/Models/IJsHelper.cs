namespace QueryWeb.Models;

public interface IJsHelper
{
    Task InvokeVoidWithObjectAsync(string method, object obj);
    Task<bool> ShowMessageAsync(string message);
}