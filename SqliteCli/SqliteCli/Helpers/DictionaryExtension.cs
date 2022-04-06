using T1.Standard.Common;
using T1.Standard.DynamicCode;

namespace SqliteCli.Repos
{
	public static class DictionaryExtension
	{
		public static T ConvertToObject<T>(this IDictionary<string, object> dict)
			where T : class, new()
		{
			var obj = new T();
			var clazz = ReflectionClass.Reflection(typeof(T));
			foreach (var key in dict.Keys)
			{
				var propName = key.Substring(0, 1).ToUpper();
				if (key.Length > 1)
				{
					propName += key.Substring(1);
				}
				var value = dict[key];
				if (!clazz.Properties.TryGetValue(propName, out var prop))
				{
					continue;
				}
				if (value == null)
				{
					continue;
				}
				var propType = prop.PropertyType;
				if (propType != value.GetType())
				{
					var propValue = value.ChangeType(propType);
					prop.Setter(obj, propValue);
				}
				else
				{
					prop.Setter(obj, value);
				}
			}
			return obj;
		}
	}
}
