using AutoMapper;

namespace MockApiWeb.Models.Services;

public static class ServiceMappingsExtension
{
    public static void AddAutoMappings(this IServiceCollection services)
    {
        var config = new MapperConfiguration(cfg => 
            cfg.AddProfile<TypeAutoMappings>());
        services.AddSingleton<IMapper>(sp => config.CreateMapper());
    }
}