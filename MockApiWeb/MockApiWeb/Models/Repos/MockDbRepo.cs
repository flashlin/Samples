using System.Text.Json;
using System.Web;
using MockApiWeb.Models.Requests;

namespace MockApiWeb.Models.Repos;

public class MockDbRepo : IMockDbRepo
{
    private readonly MockDbContext _mockDbContext;

    public MockDbRepo(MockDbContext mockDbContext)
    {
        _mockDbContext = mockDbContext;
    }

    public WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiRequest req)
    {
        return _mockDbContext.WebApiFuncInfos
            .FirstOrDefault(x => x.ProductName == req.ProductName
                                 && x.ControllerName == req.ControllerName
                                 && x.ActionName == req.ActionName, new WebApiFuncInfoEntity()
            {
                ProductName = req.ProductName,
                ControllerName = req.ControllerName,
                ActionName = req.ActionName,
                ResponseContent = GetRequestJsonContent(req),
                ResponseStatus = 200
            });
    }

    private string GetRequestJsonContent(MockWebApiRequest req)
    {
        if (!string.IsNullOrEmpty(req.RequestBody))
        {
            return req.RequestBody;
        }

        if (!string.IsNullOrEmpty(req.RequestQueryString))
        {
            var nameValues = HttpUtility.ParseQueryString(req.RequestQueryString);
            var dictionary = nameValues.AllKeys
                .Select(key => key!)
                .ToDictionary(key => key, key => nameValues[key]);
            return JsonSerializer.Serialize(dictionary);
        }

        return string.Empty;
    }
}