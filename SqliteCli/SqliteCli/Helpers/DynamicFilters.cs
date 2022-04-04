using Microsoft.EntityFrameworkCore;

namespace SqliteCli.Helpers
{
	public class DynamicFilters<T>
		where T : class
	{
		private readonly DbContext _context;

		public DynamicFilters(DbContext context)
		{
			_context = context;
		}

		public IEnumerable<T> Filter(IEnumerable<QueryableFilter> queryableFilters = null)
		{
			IQueryable<T> mainQuery = _context.Set<T>().AsQueryable().AsNoTracking();

			foreach (var filter in queryableFilters ?? new List<QueryableFilter>())
			{
				mainQuery = mainQuery.BuildExpression(_context, filter.Name, filter.Value, filter.Compare);
			}

			//mainQuery = mainQuery.OrderBy(x => x.Id);

			return mainQuery.ToList();
		}
	}
}
