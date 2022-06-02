using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;

namespace PizzaWeb.Models;

public class TemplateVariableConverter : ValueConverter<List<TemplateVariable>, string>
{
    public TemplateVariableConverter() : base(SerializeFn(),DeserializeFn())
    {
    }

    private static Expression<Func<List<TemplateVariable>, string>> SerializeFn()
    {
        var fn = (List<TemplateVariable> model) =>
        {
            var dict = model.ToDictionary(x => x.VarName);
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return jsonConvert.Serialize(dict);
        };
        return (model) => fn(model);
    }

    private static Expression<Func<string, List<TemplateVariable>>> DeserializeFn()
    {
        var fn = (string jsonStr) => jsonStr.ToTemplateVariablesList();
        return (jsonStr) => fn(jsonStr);
    }
}