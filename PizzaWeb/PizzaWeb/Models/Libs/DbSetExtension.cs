using Microsoft.EntityFrameworkCore;
using System.Linq.Expressions;
using T1.Standard.Expressions;

namespace PizzaWeb.Models.Libs
{
	public static class DbSetExtension
	{
		public static UpdateOrm<T> Set<T, TValue>(this DbSet<T> dbSet, Expression<Func<T, TValue>> memberLamda, TValue value)
			where T : class
		{
			return new UpdateOrm<T>(dbSet).Set(memberLamda, value);
		}
	}

	public class UpdateOrm<T>
		where T: class
	{
		private DbSet<T> dbSet;
		private Dictionary<string, object> _setFields = new Dictionary<string, object>();

		public UpdateOrm(DbSet<T> dbSet)
		{
			this.dbSet = dbSet;
		}

		public UpdateOrm<T> Set<TValue>(Expression<Func<T, TValue>> memberLamda, TValue value)
		{
			var simpleProperty = memberLamda.GetSimplePropertyAccess()
				.First();
			var name = simpleProperty.Name;
			_setFields[name] = value;
			return this;
		}

		public UpdateOrm<T> Where(Expression<Func<T, bool>> filterAction)
		{
			return this;	
		}

		public void Update()
		{
			
		}
	}
}
