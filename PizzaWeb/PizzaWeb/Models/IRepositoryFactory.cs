using Microsoft.EntityFrameworkCore;

namespace PizzaWeb.Models
{
	public interface IRepositoryFactory
	{
		T BuildRepository<T>()
        where T : DbContext;
	}
}