namespace QueryWeb.Models;

public interface IJsHelper
{
    Task InvokeVoidWithObjectAsync(string method, object obj, object obj1=null);
    Task<bool> ShowMessageAsync(string message);
}