using System.Text.Json.Serialization;
using LinqKit;
using Microsoft.EntityFrameworkCore;
using MockApiWeb.Controllers;
using MockApiWeb.Models.DataConstraints;
using MockApiWeb.Models.Parameters;

namespace MockApiWeb.Models.Repos;

public class MockDbContext : DbContext, IMockDbRepo
{
    public MockDbContext(DbContextOptions<MockDbContext> options)
        : base(options)
    {
    }

    public DbSet<WebApiMockInfoEntity> WebApiMockInfos { get; set; } = null!;

    public WebApiMockInfoEntity GetWebApiResponseSetting(MockWebApiParameters req)
    {
        var data = WebApiMockInfos.AsNoTracking()
            .FirstOrDefault(x => x.ProductName == req.ProductName
                                 && x.ControllerName == req.ControllerName
                                 && x.ActionName == req.ActionName);

        if (data != null)
        {
            return data;
        }

        return new WebApiMockInfoEntity()
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
        WebApiMockInfos.Add(new WebApiMockInfoEntity
        {
            ProductName = req.ProductName,
            ControllerName = req.ControllerName,
            ActionName = req.ActionName,
            ResponseContent = req.ResponseContent,
            ResponseStatus = req.ResponseStatusCode
        });
        SaveChanges();
    }

    public DefaultResponsePageData QueryDefaultResponsePage(GetWebApiSimpleSettingRequest req)
    {
        var predicate = CreatePredicate(req, req.QueryForm);

        var pageData = WebApiMockInfos.AsExpandable()
            .Where(predicate)
            .Take(req.PageSize + 1).ToList();

        return new DefaultResponsePageData
        {
            PageData = pageData,
            HasNextPage = pageData.Count > req.PageSize
        };
    }

    public int QueryDefaultResponsePageCount(GetWebApiSimpleSettingRequest req)
    {
        var predicate = CreatePredicate(req, req.QueryForm);

        var dataCount = WebApiMockInfos.AsExpandable()
            .Where(predicate)
            .Count() + req.PageSize - 1;

        return dataCount / req.PageSize;
    }

    private static ExpressionStarter<WebApiMockInfoEntity> CreatePredicate(GetWebApiSimpleSettingRequest req,
        QueryWebApiSimpleSettingForm queryForm)
    {
        var predicate = PredicateBuilder.New<WebApiMockInfoEntity>(true);
        if (!string.IsNullOrEmpty(queryForm.ProductName))
        {
            predicate.And(x => x.ProductName.Contains(queryForm.ProductName));
        }

        if (!string.IsNullOrEmpty(queryForm.ControllerName))
        {
            predicate.And(x => x.ControllerName.Contains(queryForm.ControllerName));
        }

        if (!string.IsNullOrEmpty(queryForm.ActionName))
        {
            predicate.And(x => x.ActionName.Contains(queryForm.ActionName));
        }

        predicate.And(x => x.Id > req.StartId);
        return predicate;
    }
}