using AutoMapper;
using MockApiWeb.Models.DataConstraints;
using MockApiWeb.Models.Parameters;

namespace MockApiWeb.Models.Services;

public class TypeAutoMappings : Profile
{
    public TypeAutoMappings()
    {
        CreateMap<MockWebApiRequest, MockWebApiParameters>();
        CreateMap<MockWebApiSimpleSettingRequest, MockWebApiSimpleSettingParameters>();
    }
}