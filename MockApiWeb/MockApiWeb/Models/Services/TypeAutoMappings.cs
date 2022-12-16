using AutoMapper;
using MockApiWeb.Models.DataObjects;

namespace MockApiWeb.Models.Services;

public class TypeAutoMappings : Profile
{
    public TypeAutoMappings()
    {
        CreateMap<MockWebApiRequest, MockWebApiParameters>();
    }
}