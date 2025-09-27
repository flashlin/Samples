using MassTransit;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace CodeBoyLib.MQ;

public static class MqStartup
{
    public static void AddMqService(IServiceCollection services, IHostEnvironment environment)
    {
        services.AddSingleton<MassTransitProgressQueue>();
        
        services.AddMassTransit(x =>
        {
            x.AddConsumer<JobProgressConsumer>();
            x.AddConsumer<StartJobConsumer>();

            if (environment.IsDevelopment())
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
}
