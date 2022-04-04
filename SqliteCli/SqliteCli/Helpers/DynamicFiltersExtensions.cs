using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using System.Linq.Expressions;

namespace SqliteCli.Helpers
{
	public static class DynamicFiltersExtensions
	{
		public static IQueryable<T> BuildExpression<T>(this IQueryable<T> source,
				DbContext context, string columnName, object value,
				QueryableFilterCompareEnum? compare = QueryableFilterCompareEnum.Equal)
		{
			var param = Expression.Parameter(typeof(T));

			// Get the field/column from the Entity that matches the supplied columnName value
			// If the field/column does not exists on the Entity, throw an exception; There is nothing more that can be done
			MemberExpression dataField;

			var model = context.Model.FindEntityType(typeof(T)); // start with our own entity
			var props = model.GetPropertyAccessors(param); // get all available field names including navigations
			var reference = props.First(p => Microsoft.EntityFrameworkCore.RelationalPropertyExtensions.GetColumnName(p.Item1) == columnName); // find the filtered column - you might need to handle cases where column does not exist

			dataField = reference.Item2 as MemberExpression; // we happen to already have correct property accessors in our Tuples	

			//ConstantExpression constant = !string.IsNullOrWhiteSpace(value)
			//	? Expression.Constant(value.Trim(), typeof(string))
			//	: Expression.Constant(value, typeof(string));


			var constant = Expression.Constant(value, value.GetType());
			var binary = GetBinaryExpression(dataField, constant, compare);
			var lambda = (Expression<Func<T, bool>>)Expression.Lambda(binary, param);
			return source.Where(lambda);
		}

		private static IEnumerable<Tuple<IProperty, Expression>> GetPropertyAccessors(this IEntityType model, Expression param)
		{
			var result = new List<Tuple<IProperty, Expression>>();

			result.AddRange(model.GetProperties()
										.Where(p => !p.IsShadowProperty()) // this is your chance to ensure property is actually declared on the type before you attempt building Expression
										.Select(p => new Tuple<IProperty, Expression>(p, Expression.Property(param, p.Name)))); // Tuple is a bit clunky but hopefully conveys the idea

			foreach (var nav in model.GetNavigations().Where(p => p is Navigation))
			{
				var parentAccessor = Expression.Property(param, nav.Name); // define a starting point so following properties would hang off there
				result.AddRange(GetPropertyAccessors(nav.ForeignKey.PrincipalEntityType, parentAccessor)); //recursively call ourselves to travel up the navigation hierarchy
			}

			return result;
		}

		private static BinaryExpression GetBinaryExpression(MemberExpression member,
			ConstantExpression constant, QueryableFilterCompareEnum? comparisonOperation)
		{
			switch (comparisonOperation)
			{
				case QueryableFilterCompareEnum.NotEqual:
					return Expression.Equal(member, constant);
				case QueryableFilterCompareEnum.GreaterThan:
					return Expression.GreaterThan(member, constant);
				case QueryableFilterCompareEnum.GreaterThanOrEqual:
					return Expression.GreaterThanOrEqual(member, constant);
				case QueryableFilterCompareEnum.LessThan:
					return Expression.LessThan(member, constant);
				case QueryableFilterCompareEnum.LessThanOrEqual:
					return Expression.LessThanOrEqual(member, constant);
				case QueryableFilterCompareEnum.Equal:
				default:
					return Expression.Equal(member, constant);
			}
		}
	}
}
