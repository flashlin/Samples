using System.Text.Json;
using System.Web;

namespace MockApiWeb.Models.Dtos;

public class MockWebApiRequest
{
    public string ProductName { get; set; } = string.Empty;
    public string RequestBody { get; set; } = string.Empty;
    public string ControllerName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public string RequestQueryString { get; set; } = string.Empty;
}

public class MockWebApiParameters
{
    public string ProductName { get; set; } = string.Empty;
    public string RequestBody { get; set; } = string.Empty;
    public string ControllerName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public string RequestQueryString { get; set; } = string.Empty;

    public string GetRequestJsonContent()
    {
        if (!string.IsNullOrEmpty(RequestBody))
        {
            return RequestBody;
        }

        if (!string.IsNullOrEmpty(RequestQueryString))
        {
            var nameValues = HttpUtility.ParseQueryString(RequestQueryString);
            var dictionary = nameValues.AllKeys
                .Select(key => key!)
                .ToDictionary(key => key, key => nameValues[key]);
            return JsonSerializer.Serialize(dictionary);
        }

        return string.Empty;
    }
}
