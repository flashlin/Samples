namespace MockApiWeb.Models.SwaggerUtils;

public class SwaggerPath
{
    public string ApiUrl { get; set; } = string.Empty;
    public string AccessMethod { get; set; } = string.Empty;
    public string[] Tags { get; set; } = Array.Empty<string>();
    public List<SwaggerParameter> Parameters { get; set; } = new();
}