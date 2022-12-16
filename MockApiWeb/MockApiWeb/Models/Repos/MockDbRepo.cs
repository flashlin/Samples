using System.Text.Json;
using System.Web;
using Microsoft.EntityFrameworkCore;
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
        var data = _mockDbContext.WebApiFuncInfos.AsNoTracking()
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

    public void AddMockWebApiSimpleSetting(MockWebApiSimpleSettingParameters req)
    {
        _mockDbContext.WebApiFuncInfos.Add(new WebApiFuncInfoEntity
        {
            ProductName = req.ProductName,
            ControllerName = req.ControllerName,
            ActionName = req.ActionName,
            ResponseContent = req.ResponseContent,
            ResponseStatus = req.ResponseStatusCode
        });
        _mockDbContext.SaveChanges();
    }
}