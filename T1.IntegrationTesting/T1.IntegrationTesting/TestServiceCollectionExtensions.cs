using Microsoft.Extensions.DependencyInjection;

namespace T1.IntegrationTesting;

public static class TestServiceCollectionExtensions
{
    public static IServiceCollection ReplaceWithMock<TService, TMock>(
        this IServiceCollection services)
        where TMock : class, TService
        where TService : class
    {
        // 1. Find all related service registrations
        var relatedDescriptors = services
            .Select((descriptor, index) => new { descriptor, index })
            .Where(x => IsRelatedToService<TService>(x.descriptor))
            .ToList();

        if (!relatedDescriptors.Any())
        {
            // No registration found, use default method
            services.AddSingleton<TService, TMock>();
            return services;
        }

        // 2. Keep original registration info
        var primaryDescriptor = relatedDescriptors.First().descriptor;
        var lifetime = primaryDescriptor.Lifetime;
        
        // 3. Check if it's Decorator pattern
        bool isDecorator = relatedDescriptors.Count > 1;
        
        // 4. Check if it's Named/Keyed service (ASP.NET Core 8+)
        bool isKeyed = primaryDescriptor.ServiceKey != null;

        // 5. Remove all related registrations
        foreach (var item in relatedDescriptors.OrderByDescending(x => x.index))
        {
            services.RemoveAt(item.index);
        }

        // 6. Register Mock based on original registration method
        if (isKeyed)
        {
            // Named/Keyed service
            services.Add(new ServiceDescriptor(
                typeof(TService),
                primaryDescriptor.ServiceKey,
                typeof(TMock),
                lifetime));
        }
        else if (isDecorator)
        {
            // Decorator pattern - only register outermost layer
            services.Add(new ServiceDescriptor(
                typeof(TService),
                typeof(TMock),
                lifetime));
        }
        else
        {
            // Normal registration
            services.Add(new ServiceDescriptor(
                typeof(TService),
                typeof(TMock),
                lifetime));
        }

        return services;
    }

    public static IServiceCollection ReplaceWithMock(
        this IServiceCollection services,
        Type serviceType,
        object mockInstance)
    {
        // 1. Find all related service registrations
        var relatedDescriptors = services
            .Select((descriptor, index) => new { descriptor, index })
            .Where(x => IsRelatedToService(x.descriptor, serviceType))
            .ToList();

        if (!relatedDescriptors.Any())
        {
            // No registration found, use default method
            services.AddSingleton(serviceType, mockInstance);
            return services;
        }

        // 2. Keep original registration info
        var primaryDescriptor = relatedDescriptors.First().descriptor;
        var lifetime = primaryDescriptor.Lifetime;
        
        // 3. Check if it's Decorator pattern
        bool isDecorator = relatedDescriptors.Count > 1;
        
        // 4. Check if it's Named/Keyed service (ASP.NET Core 8+)
        bool isKeyed = primaryDescriptor.ServiceKey != null;

        // 5. Remove all related registrations
        foreach (var item in relatedDescriptors.OrderByDescending(x => x.index))
        {
            services.RemoveAt(item.index);
        }

        // 6. Register Mock based on original registration method
        if (isKeyed)
        {
            // Named/Keyed service
            services.Add(new ServiceDescriptor(
                serviceType,
                primaryDescriptor.ServiceKey,
                mockInstance));
        }
        else
        {
            // Normal registration or Decorator pattern
            services.Add(new ServiceDescriptor(
                serviceType,
                mockInstance));
        }

        return services;
    }

    private static bool IsRelatedToService<TService>(ServiceDescriptor descriptor)
    {
        var serviceType = typeof(TService);
        
        return descriptor.ServiceType == serviceType ||
               (descriptor.ImplementationType != null && 
                serviceType.IsAssignableFrom(descriptor.ImplementationType)) ||
               (descriptor.ImplementationFactory != null && 
                descriptor.ServiceType.IsAssignableFrom(serviceType)) ||
               (descriptor.ImplementationInstance != null && 
                descriptor.ImplementationInstance is TService);
    }

    private static bool IsRelatedToService(ServiceDescriptor descriptor, Type serviceType)
    {
        return descriptor.ServiceType == serviceType ||
               (descriptor.ImplementationType != null && 
                serviceType.IsAssignableFrom(descriptor.ImplementationType)) ||
               (descriptor.ImplementationFactory != null && 
                descriptor.ServiceType.IsAssignableFrom(serviceType)) ||
               (descriptor.ImplementationInstance != null && 
                serviceType.IsInstanceOfType(descriptor.ImplementationInstance));
    }
}