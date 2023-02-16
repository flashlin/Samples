using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalApiController : ControllerBase
{
    [HttpPost]
    public OkResult Knock(KnockRequest req)
    {
        return Ok();
    }
}

public class KnockRequest
{
    public string UniqueId { get; set; } = null!;
}

public interface IQueryClient
{
    Task EchoAsync(ILocalEnvironment localEnvironment);
}

public class QueryClient : IQueryClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public QueryClient(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
        _baseUrl = "http://example.com/api/";
    }

    public async Task EchoAsync(ILocalEnvironment localEnvironment)
    {
        await PostJsonVoidAsync("echo", localEnvironment);
    }

    private async Task<T?> PostJsonAsync<T>(string relativeUrl, object request)
    {
        var response = await PostAsync(relativeUrl, request);
        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<T>(responseJson);
    }

    private async Task<HttpResponseMessage> PostAsync(string relativeUrl, object request)
    {
        var json = JsonSerializer.Serialize(request);
        var requestContent = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"{_baseUrl}{relativeUrl}", requestContent);
        response.EnsureSuccessStatusCode();
        return response;
    }


    private async Task PostJsonVoidAsync(string relativeUrl, object request)
    {
        _ = await PostAsync(relativeUrl, request);
    }
}