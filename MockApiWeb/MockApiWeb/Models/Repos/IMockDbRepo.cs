using MockApiWeb.Models.Requests;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiRequest req);
}