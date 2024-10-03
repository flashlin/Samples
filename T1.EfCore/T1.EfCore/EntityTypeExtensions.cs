using System.Linq.Expressions;
using System.Reflection;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class EntityTypeExtensions
{
    public static List<IProperty> GenerateMatchCondition<TEntity>(this IEntityType entityType,
        Expression<Func<TEntity, object>> matchExpression)
        where TEntity : class
    {
        if (matchExpression.Body is MemberExpression memberExpression)
        {
            if (typeof(TEntity) != memberExpression.Expression?.Type || memberExpression.Member is not PropertyInfo)
                throw new InvalidOperationException("MatchColumnsHaveToBePropertiesOfTheTEntityClass");
            var property = entityType.FindProperty(memberExpression.Member.Name);
            if (property == null)
                throw new InvalidOperationException("UnknownProperty memberExpression.Member.Name");
            return [property];
        }

        if (matchExpression.Body is UnaryExpression unaryExpression)
        {
            if (unaryExpression.Operand is not MemberExpression memberExp || memberExp.Member is not PropertyInfo ||
                typeof(TEntity) != memberExp.Expression?.Type)
                throw new InvalidOperationException("MatchColumnsHaveToBePropertiesOfTheTEntityClass");
            var property = entityType.FindProperty(memberExp.Member.Name);
            if (property == null)
                throw new InvalidOperationException("UnknownProperty, memberExp.Member.Name");
            return [property];
        }
        
        if (matchExpression.Body is NewExpression newExpression)
        {
            var joinColumns = new List<IProperty>();
            foreach (var expression in newExpression.Arguments)
            {
                var arg = (MemberExpression)expression;
                if (arg is not { Member: PropertyInfo } || typeof(TEntity) != arg.Expression?.Type)
                    throw new InvalidOperationException("MatchColumns Have To Be Properties Of The EntityClass");
                var property = entityType.FindProperty(arg.Member.Name);
                if (property == null)
                    throw new InvalidOperationException($"UnknownProperty {arg.Member.Name}");
                joinColumns.Add(property);
            }
            return joinColumns;
        }

        throw new ArgumentException("Unsupported where expression");
    }
}