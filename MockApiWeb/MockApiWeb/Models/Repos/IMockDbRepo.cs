using MockApiWeb.Models.Parameters;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiMockInfoEntity GetWebApiResponseSetting(MockWebApiParameters req);
    void AddMockWebApiSimpleSetting(MockWebApiSimpleSettingParameters req);
}