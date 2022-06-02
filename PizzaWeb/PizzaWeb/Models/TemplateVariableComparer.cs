using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;

namespace PizzaWeb.Models;

public class TemplateVariableComparer : ValueComparer<List<TemplateVariable>>
{
    public TemplateVariableComparer()
        : base(EqualsFn(), HashCodeFn(), SnapshotFn())
    {
    }

    private static Expression<Func<List<TemplateVariable>?, List<TemplateVariable>?, bool>> EqualsFn()
    {
        var fn = (List<TemplateVariable>? left, List<TemplateVariable>? right) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            var leftJson = jsonConvert.Serialize(left);
            var rightJson = jsonConvert.Serialize(right);
            return leftJson == rightJson;
        };
        return (left, right) => fn(left, right);
    }

    private static Expression<Func<List<TemplateVariable>, int>> HashCodeFn()
    {
        var fn = (List<TemplateVariable>? model) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return model == null ? 0 : jsonConvert.Serialize(model).GetHashCode();
        };
        return (model) => fn(model);
    }

    private static Expression<Func<List<TemplateVariable>, List<TemplateVariable>>> SnapshotFn()
    {
        var fn = (List<TemplateVariable> model) =>
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return jsonConvert.Deserialize<List<TemplateVariable>>(jsonConvert.Serialize(model));
        };
        return (model) => fn(model);
    }
}