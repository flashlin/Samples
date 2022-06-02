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

public class VariableOptionListConverter : ValueConverter<List<VariableOption>, string>
{
    public VariableOptionListConverter() : 
        base(SerializeFn(),DeserializeFn())
    {
    }

    private static Expression<Func<List<VariableOption>, string>> SerializeFn()
    {
        var fn = (List<VariableOption> model) =>
        {
            var dict = model.ToDictionary(x => x.VarName);
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            return jsonConvert.Serialize(dict);
        };
        return (model) => fn(model);
    }

    private static Expression<Func<string, List<VariableOption>>> DeserializeFn()
    {
        var fn = (string jsonStr) => jsonStr.ToVariableOptionsList();
        return (jsonStr) => fn(jsonStr);
    }
}
