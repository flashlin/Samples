using System.Text.Json;
using System.Web;
using MockApiWeb.Models.DataObjects;

namespace MockApiWeb.Models.Repos;

public class MockDbRepo : IMockDbRepo
{
    private readonly MockDbContext _mockDbContext;

    public MockDbRepo(MockDbContext mockDbContext)
    {
        _mockDbContext = mockDbContext;
    }

    public WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiParameters req)
    {
        var data = _mockDbContext.WebApiFuncInfos
            .FirstOrDefault(x => x.ProductName == req.ProductName
                                 && x.ControllerName == req.ControllerName
                                 && x.ActionName == req.ActionName);

        if (data != null)
        {
            return data;
        }

        return new WebApiFuncInfoEntity()
        {
            ProductName = req.ProductName,
            ControllerName = req.ControllerName,
            ActionName = req.ActionName,
            ResponseContent = req.GetRequestJsonContent(),
            ResponseStatus = 200
        };
    }
}