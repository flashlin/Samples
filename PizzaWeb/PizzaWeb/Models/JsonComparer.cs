using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;

namespace PizzaWeb.Models;

public class JsonComparer<T> : ValueComparer<T>
{
    public JsonComparer()
        : base(EqualsFn(), HashCodeFn(), SnapshotFn())
    {
    }

    private static Expression<Func<T?, T?, bool>> EqualsFn()
    {
        var fn = (T? left, T? right) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            var leftJson = (left != null) ? jsonConvert.Serialize(left) : String.Empty;
            var rightJson = (right != null) ? jsonConvert.Serialize(right) : String.Empty;
            return leftJson == rightJson;
        };
        return (left, right) => fn(left, right);
    }

    private static Expression<Func<T, int>> HashCodeFn()
    {
        var fn = (T? model) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return model == null ? 0 : jsonConvert.Serialize(model).GetHashCode();
        };
        return (model) => fn(model);
    }

    private static Expression<Func<T, T>> SnapshotFn()
    {
        var fn = (T model) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return jsonConvert.Deserialize<T>(jsonConvert.Serialize(model));
        };
        return (model) => fn(model);
    }
}