using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Mvc;
using MockApiWeb.Models.DataObjects;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MgmtApiController : ControllerBase
{
    [HttpPost]
    public JsonResult CreateDefaultResponse(MockWebApiRequest req)
    {
        return new JsonResult(new
        {
            Id = 133,
            Name = "Flashcc"
        });
    }
}

public class SwaggerJsonFetcher
{
    private HttpClient _httpClient;

    public SwaggerJsonFetcher(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
        //httpClient.BaseAddress = new Uri("https://api.line.me");
        //httpClient.DefaultRequestHeaders.Add("authorization", "Bearer {CannelAccessToken}");
    }

    public async Task<SwaggerJsonDocument> Read(string swaggerJsonUrl)
    {
        var url = "https://localhost:44325/swagger/v1/swagger.json";
        var response = await _httpClient.GetAsync(swaggerJsonUrl);
        var content = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<SwaggerJsonDocument>(content) ?? new SwaggerJsonDocument();
    }
}

public class SwaggerJsonDocument
{
    public string OpenApi { get; set; } = "3.0.1";
    public SwaggerInfo Info { get; set; } = new();
    public SwaggerPaths Paths { get; set; } = new();
}

public class SwaggerPaths
{
}

public class SwaggerPath
{
    public string ApiUrl { get; set; } = string.Empty;
    public string AccessMethod { get; set; } = string.Empty;
    public string[] Tags { get; set; } = Array.Empty<string>();
    public List<SwaggerParameter> Parameters { get; set; } = new();
}

public class SwaggerParameterConverter : JsonConverter<SwaggerParameter>
{
    public override bool CanConvert(Type typeToConvert)
    {
        return true;
    }

    public override SwaggerParameter? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        throw new NotImplementedException();
    }

    public override void Write(Utf8JsonWriter writer, SwaggerParameter value, JsonSerializerOptions options)
    {
        throw new NotImplementedException();
    }
}

public class SwaggerParameter
{
    public string Name { get; set; } = string.Empty;
    public string SwaggerDataType { get; set; } = "integer";
    public string Format { get; set; } = "int32";
}

public class SwaggerInfo
{
    public string Title { get; set; } = string.Empty;
    public string Version { get; set; } = "1.0";
}