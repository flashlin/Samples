using MockApiWeb.Models.DataObjects;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiParameters req);
}