using Microsoft.EntityFrameworkCore;

namespace PizzaWeb.Models
{
	public interface IDbContextOptionsFactory
	{
		DbContextOptions Create();
	}
}