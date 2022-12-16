using MockApiWeb.Models.Dtos;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiParameters req);
}