using MassTransit;
using Microsoft.Extensions.DependencyInjection;

namespace CodeBoyLib.MQ;

public static class MqStartup
{
    public static void AddMqService(IServiceCollection services, bool isDevelopment = false)
    {
        services.AddSingleton<MassTransitProgressQueue>();
        
        services.AddMassTransit(x =>
        {
            x.AddConsumer<JobProgressConsumer>();
            x.AddConsumer<StartJobConsumer>();

            if (isDevelopment)
            {
                // Use InMemory transport (single machine, no RabbitMQ Server required)
                x.UsingInMemory((context, cfg) =>
                {
                    cfg.ConfigureEndpoints(context);
                });
            }
            else
            {
                // Use RabbitMQ for production environment
                x.UsingRabbitMq((context, cfg) =>
                {
                    cfg.Host("localhost", "/", h =>
                    {
                        h.Username("guest");
                        h.Password("guest");
                    });

                    cfg.ConfigureEndpoints(context);
                });
            }
        });
    }

    public static void AddMqServiceInMemory(IServiceCollection services)
    {
        AddMqService(services, isDevelopment: true);
    }

    public static void AddMqServiceRabbitMq(IServiceCollection services, string host = "localhost", 
        string username = "guest", string password = "guest", string virtualHost = "/")
    {
        services.AddSingleton<MassTransitProgressQueue>();
        
        services.AddMassTransit(x =>
        {
            x.AddConsumer<JobProgressConsumer>();
            x.AddConsumer<StartJobConsumer>();

            x.UsingRabbitMq((context, cfg) =>
            {
                cfg.Host(host, virtualHost, h =>
                {
                    h.Username(username);
                    h.Password(password);
                });

                cfg.ConfigureEndpoints(context);
            });
        });
    }
}