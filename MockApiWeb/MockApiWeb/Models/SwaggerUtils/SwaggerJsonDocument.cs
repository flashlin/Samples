namespace MockApiWeb.Models.SwaggerUtils;

public class SwaggerJsonDocument
{
    public string OpenApi { get; set; } = "3.0.1";
    public SwaggerInfo Info { get; set; } = new();
    public SwaggerPaths Paths { get; set; } = new();
}