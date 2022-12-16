using MockApiWeb.Models.Parameters;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiFuncInfoEntity GetWebApiResponseSetting(MockWebApiParameters req);
    void AddMockWebApiSimpleSetting(MockWebApiSimpleSettingParameters req);
}