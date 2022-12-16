using AutoMapper;
using MockApiWeb.Models.Dtos;

namespace MockApiWeb.Models.Services;

public class TypeAutoMappings : Profile
{
    public TypeAutoMappings()
    {
        CreateMap<MockWebApiRequest, MockWebApiParameters>();
    }
}