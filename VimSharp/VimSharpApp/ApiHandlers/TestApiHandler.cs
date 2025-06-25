using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

public interface ITestApiHandler
{
    Task<JsonExampleResponse> Test();
}

public class JsonExampleResponse
{
    public int UserId { get; set; }
    public int Id { get; set; }
    public string Title { get; set; }
    public string Body { get; set; }
}


public class TestApiHandler : ITestApiHandler
{
    public async Task<JsonExampleResponse> Test()
    {
        var httpClient = new HttpClient();
        var response = await httpClient.GetFromJsonAsync<JsonExampleResponse>("https://jsonplaceholder.typicode.com/posts/1");
        return response!;
    }

    public static void MapEndpoints(WebApplication app)
    {
        app.MapPost("/api/Test", (ITestApiHandler handler) => handler.Test());
    }
}
