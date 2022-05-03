using Microsoft.Extensions.Hosting;

namespace SqliteCli.Factories;

public class HostFactory
{
    public IHostBuilder Create(string[] args)
    {
        if (args.Contains("-web"))
        {
            return MinimalHostingApi.CreateBuilder(args);
        }

        return Host.CreateDefaultBuilder(args);
    }
}