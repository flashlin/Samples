using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

public interface ITestApiHandler
{
    Task<string> Test();
}

public class TestApiHandler : ITestApiHandler
{
    public Task<string> Test()
    {
        return Task.FromResult("Hello, World!");
    }

    public static void MapEndpoints(WebApplication app)
    {
        app.MapPost("/api/Test", (ITestApiHandler handler) => handler.Test());
    }
}
