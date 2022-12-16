namespace MockApiWeb.Models.DataConstraints;

public class MockWebApiRequest
{
    public string ProductName { get; set; } = string.Empty;
    public string RequestBody { get; set; } = string.Empty;
    public string ControllerName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public string RequestQueryString { get; set; } = string.Empty;
}