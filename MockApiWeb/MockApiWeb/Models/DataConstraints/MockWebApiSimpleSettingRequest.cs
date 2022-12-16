namespace MockApiWeb.Models.DataConstraints;

public class MockWebApiSimpleSettingRequest
{
    public string ProductName { get; set; } = string.Empty;
    public string ControllerName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public string ResponseContent { get; set; } = string.Empty;
    public int ResponseStatusCode { get; set; } = 200;
}