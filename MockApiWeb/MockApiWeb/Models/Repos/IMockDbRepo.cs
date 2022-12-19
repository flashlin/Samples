using MockApiWeb.Models.DataConstraints;
using MockApiWeb.Models.Parameters;

namespace MockApiWeb.Models.Repos;

public interface IMockDbRepo
{
    WebApiMockInfoEntity GetWebApiResponseSetting(MockWebApiParameters req);
    void AddMockWebApiSimpleSetting(MockWebApiSimpleSettingParameters req);
    DefaultResponsePageData QueryDefaultResponsePage(GetWebApiSimpleSettingRequest req);
    int QueryDefaultResponsePageCount(GetWebApiSimpleSettingRequest req);
}