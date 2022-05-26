using Microsoft.EntityFrameworkCore;

namespace PizzaWeb.Models;

public class RepositoryFactory : IRepositoryFactory
{
    private readonly IServiceProvider _serviceProvider;

    public RepositoryFactory(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public T BuildRepository<T>()
        where T : DbContext
    {
        return _serviceProvider.GetService<IDbContextFactory<T>>()
            .CreateDbContext();
    }
}