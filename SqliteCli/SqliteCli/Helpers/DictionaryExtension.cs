using System.Reflection;
using T1.Standard.Common;
using T1.Standard.DynamicCode;

namespace SqliteCli.Repos
{
	public static class DictionaryExtension
	{
		public static T ToSummary<T>(this IEnumerable<T> list)
			where T : class, new()
		{
			var summary = new T();
			var clazzz = ReflectionClass.Reflection(typeof(T));
			var first = true;
			foreach (var item in list)
			{
				foreach (var prop in clazzz.Properties.Values)
				{
					if (!prop.PropertyType.IsValueType)
					{
						continue;
					}

					var propValue = prop.Getter(item);
					if (first)
					{
						prop.Setter(summary, propValue);
						if (prop.PropertyType == typeof(DateTime))
						{
							prop.Setter(summary, DateTime.Now);
						}
					}
					else
					{
						object sumValue = prop.Getter(summary);
						if (prop.PropertyType == typeof(decimal))
						{
							sumValue = (decimal)sumValue + (decimal)propValue;
						}
						else if (prop.PropertyType == typeof(int))
						{
							sumValue = (int)sumValue + (int)propValue;
						}
						prop.Setter(summary, sumValue);
					}
				}
				first = false;
			}
			return summary;
		}


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
