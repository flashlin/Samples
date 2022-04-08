using Microsoft.Data.SqlClient.Server;
using System;
using System.Collections.Generic;
using System.Data;
using System.Reflection;
using System.Text;
using T1.Standard.DynamicCode;

namespace PreviewLibrary
{
	public static class ClassToSqlVariableTableExtension
	{
		public static string Dump(this SqlDataRecord record)
		{
			var sb = new StringBuilder();
			sb.Append("{");
			for (var i = 0; i < record.FieldCount; i++)
			{
				if (i != 0)
				{
					sb.Append(",");
				}
				var name = record.GetName(i);
				var value = record.GetValue(i);
				var fieldType = record.GetFieldType(i);
				if (fieldType == typeof(string))
				{
					sb.Append($"{name}:'{value}'");
				}
				else
				{
					sb.Append($"{name}:{value}");
				}
			}
			sb.Append("}");
			return sb.ToString();
		}

		public static List<SqlDataRecord> ToSqlVariableTvp(this object obj)
		{
			var dataTable = new List<SqlDataRecord>();
			var clazz = ReflectionClass.Reflection(obj.GetType());
			foreach (var prop in clazz.Properties)
			{
				var dr = new SqlDataRecord(
					new SqlMetaData("Name", SqlDbType.VarChar, 255),
					new SqlMetaData("DataType", SqlDbType.VarChar, 255),
					new SqlMetaData("DataValue", SqlDbType.Variant)
				);
				dr.SetString(0, "@" + prop.Key);
				dr.SetString(1, GetSqlDbType((PropertyInfo)prop.Value.Info));
				dr.SetValue(2, prop.Value.Getter(obj));
				dataTable.Add(dr);
			}
			return dataTable;
		}

		private static string GetSqlDbType(PropertyInfo propInfo)
		{
			var dbTypeAttr = propInfo.GetCustomAttribute<SqlDbTypeAttribute>() ?? GetSqlDbTypeAttribute(propInfo.PropertyType);
			return dbTypeAttr.DeclationDbType;
		}

		private static SqlDbTypeAttribute GetSqlDbTypeAttribute(Type type)
		{
			var typeToDbType = new Dictionary<Type, string>()
			{
				{ typeof(int), "int" },
				{ typeof(string), "nvarchar(50)" },
				{ typeof(decimal), "decimal(18,3)" },
				{ typeof(bool), "bit" },
				{ typeof(DateTime), "datetime" },
			};
			return new SqlDbTypeAttribute(typeToDbType[type]);
		}
	}

	[AttributeUsage(AttributeTargets.Property)]
	public class SqlDbTypeAttribute : Attribute
	{
		public SqlDbTypeAttribute(string declationDbType)
		{
			DeclationDbType = declationDbType;
		}

		public string DeclationDbType { get; set; }
	}
}
