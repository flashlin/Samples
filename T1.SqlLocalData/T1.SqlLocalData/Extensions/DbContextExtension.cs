﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.EntityFrameworkCore;
using T1.Standard.DynamicCode;

namespace T1.SqlLocalData.Extensions;

public static class DbContextExtension
{
	public static void CreateTable(this DbContext context, Type entityType)
	{
		var createTableSqlCode = GenerateCreateTableSqlCode(entityType);
		context.Database.ExecuteSqlRaw(createTableSqlCode);
	}

	private static string GenerateCreateTableSqlCode(Type entityType)
	{
		var columnList = GetTableColumnsInfo(entityType);
		var sb = new StringBuilder();
		sb.AppendLine($"CREATE TABLE {GetTableName(entityType)}");
		sb.AppendLine("(");
		var columnListCode =
			string.Join(",", columnList.Select(x =>
			{
				var key = x.IsKey ? "PRIMARY KEY" : "";
				return $"{x.Name} {x.DataType} {key}";
			}));
		sb.Append(columnListCode);
		sb.AppendLine(");");
		return sb.ToString();
	}

	private static string GetTableName(Type entityType)
	{
		var tableAttribute = entityType.GetCustomAttribute<TableAttribute>();
		if (tableAttribute == null)
		{
			return entityType.Name;
		}
		return tableAttribute.Name;
	}

	private static IEnumerable<ColumnInfo> GetTableColumnsInfo(Type entityType)
	{
		var entityClass = ReflectionClass.Reflection(entityType);
		foreach (var prop in entityClass.Properties)
		{
			var propName = prop.Key;
			var propType = prop.Value.PropertyType;
			var keyAttribute = prop.Value.Info.GetCustomAttribute<KeyAttribute>();
			var isKey = keyAttribute != null;

			var dataType = GetDataType(propType);

			yield return new ColumnInfo
			{
				Name = propName,
				DataType = dataType,
				IsKey = isKey
			};
		}
	}

	private static bool IsNullableType(Type type)
	{
		return Nullable.GetUnderlyingType(type) != null;
	}

	private static Type GetUnderlyingType(Type type)
	{
		var underlyingType = Nullable.GetUnderlyingType(type);
		if (underlyingType != null)
		{
			return underlyingType;
		}
		return type;
	}

	private static string GetDataType(Type propType)
	{
		var dataType = GetRawDataType(GetUnderlyingType(propType));

		if (IsNullableType(propType))
		{
			dataType += " NULL";
		}
		return dataType;
	}

	private static string GetRawDataType(Type propType)
	{
		var typeToDataTypeDict = new Dictionary<Type, string>
		{
			{ typeof(string), "VARCHAR(50)" },
			{ typeof(int), "INT" },
			{ typeof(long), "BIGINT" },
			{ typeof(DateTime), "DATETIME" },
			{ typeof(decimal), "DECIMAL(10,2)" },
			{ typeof(bool), "BIT" }
		};
		return typeToDataTypeDict[propType];
	}
}
