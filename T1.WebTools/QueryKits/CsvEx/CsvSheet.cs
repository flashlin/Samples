﻿using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CsvHelper;
using CsvHelper.Configuration;
using T1.Standard.IO;

namespace QueryKits.CsvEx;

public class CsvSheet
{
   public static CsvSheet ReadFromStream(Stream csvStream, string delimiter)
   {
      using var textReader = new StreamReader(csvStream);
      return ReadFromTextReader(textReader, delimiter);
   }

   public static CsvSheet ReadFromString(string csvData)
   {
      var delimiterResult = ParseCsvDelimiter(csvData);
      if (delimiterResult.Delimiter == " ")
      {
         csvData = csvData.Replace(" ", "\t");
         delimiterResult.Delimiter = "\t";
      }
      using var strReader = new StringReader(csvData);
      return ReadFromTextReader(strReader, delimiterResult.Delimiter);
   }
   
   public static CsvSheet ReadFromTextReader(TextReader textReader, string delimiter)
   {
      var option = new CsvConfiguration(CultureInfo.InvariantCulture)
      {
         Delimiter = delimiter,
         Mode = CsvMode.RFC4180,
         Encoding = Encoding.UTF8,
         HasHeaderRecord = true,
      };
      using var csvReader = new CsvReader(textReader, option);
      csvReader.Read();
      csvReader.ReadHeader();

      var csvSheet = new CsvSheet
      {
         Delimiter = delimiter,
         Headers = csvReader.HeaderRecord!.Select(name => new CsvHeader
         {
            Name = name,
            ColumnType = ColumnType.String
         }).ToList(),
      };

      while (csvReader.Read())
      {
         var row = new Dictionary<string, string>();
         foreach (var header in csvSheet.Headers)
         {
            var value = csvReader.GetField(header.Name) ?? string.Empty;
            row[header.Name] = value;
         }
         csvSheet.Rows.Add(row);
      }
      
      csvSheet.ParseHeadersType();
      return csvSheet;
   }

   public string Delimiter { get; set; } = ",";
   public List<CsvHeader> Headers { get; init; } = new();
   
   public List<Dictionary<string, string>> Rows { get; set; } = new();

   public void ParseHeadersType()
   {
      if (Rows.Count == 0)
      {
         return;
      }

      var row = Rows[0];
      foreach (var header in Headers)
      {
         var value = row[header.Name];
         header.ColumnType = decimal.TryParse(value, out _) ? ColumnType.Number : ColumnType.String;
      }
   }

   public static CsvSheet ReadFrom(string csvFile, string delimiter)
   {
      using var stream = new FileStream(csvFile, FileMode.Open);
      return ReadFromStream(stream, delimiter);
   }

   public void SaveToFile(string file)
   {
      // var option = new CsvConfiguration(CultureInfo.InvariantCulture)
      // {
      //    Delimiter = Delimiter,
      //    Mode = CsvMode.RFC4180,
      //    Encoding = Encoding.UTF8,
      //    HasHeaderRecord = true,
      // };
      // using var stream = new FileStream(file, FileMode.Create);
      // using var streamWriter = new StreamWriter(stream, Encoding.UTF8);
      // using var csvWriter = new CsvWriter(streamWriter, option);
      // foreach (var header in Headers)
      // {
      //    csvWriter.WriteField(header.Name);
      // }
      // csvWriter.NextRecord();
      // csvWriter.WriteRecords(Rows);
      
      using var stream = new FileStream(file, FileMode.Create);
      SaveToStream(stream);
   }

   public static CsvDelimiter ParseCsvDelimiter(string text)
   {
        var sr = new StringReader(text);
        var header = sr.ReadLine();
        if (header == null)
        {
           return new CsvDelimiter
           {
              Success = false,
           };
        }

        if (header.Contains('\t'))
        {
           var tCount1 = header.Split('\t').Length;
           var row0 = sr.ReadLine()!;
           var tCount2 = row0.Split('\t').Length;
           if (tCount1 == tCount2 && tCount1 != 0)
           {
              return new CsvDelimiter
              {
                 Success = true,
                 Delimiter = "\t"
              };
           }
        }

        if (header.Contains(','))
        {
           var commaCount1 = header.Split(',').Length;
           var commaCount2 = header.Split(',').Length;
           if (commaCount1 == commaCount2 && commaCount1 != 0)
           {
              return new CsvDelimiter
              {
                 Success = true,
                 Delimiter = ","
              };
           }
        }

        if (header.Contains(' '))
        {
           var spaceCount1 = header.Split(' ').Length;
           var spaceCount2 = header.Split(' ').Length;
           if (spaceCount1 == spaceCount2 && spaceCount1 != 0)
           {
              return new CsvDelimiter()
              {
                 Success = true,
                 Delimiter = " "
              };
           }
        }

        return new CsvDelimiter
        {
           Success = false,
        };
   }

   public void SaveToStream(Stream stream)
   {
      var option = new CsvConfiguration(CultureInfo.InvariantCulture)
      {
         Delimiter = Delimiter,
         Mode = CsvMode.RFC4180,
         Encoding = Encoding.UTF8,
         HasHeaderRecord = true,
      };
      var streamWriter = new StreamWriter(stream, Encoding.UTF8);
      using var csvWriter = new CsvWriter(streamWriter, option, true);
      foreach (var header in Headers)
      {
         csvWriter.WriteField(header.Name);
      }
      csvWriter.NextRecord();
      foreach (var row in Rows)
      {
         foreach (var header in Headers)
         {
            csvWriter.WriteField(row[header.Name]);
         }
         csvWriter.NextRecord();
      }
      csvWriter.Flush();
   }

   public string SaveToString()
   {
      var memoryStream = new MemoryStream();
      SaveToStream(memoryStream);
      memoryStream.Position = 0;
      var reader = new StreamReader(memoryStream);
      return reader.ReadToEnd();
   }
}

public class CsvDelimiter
{
   public bool Success { get; set; }
   public string Delimiter { get; set; } = string.Empty;
}

public interface IMyJsonSerializer
{
   string Serialize<T>(T obj);
   T? Deserialize<T>(string json);
}

public class MyJsonSerializer : IMyJsonSerializer
{
    private static readonly JsonSerializerOptions JsonOptions = CreateDefaultSerializeOptions();
   
   public string Serialize<T>(T obj)
   {
      return JsonSerializer.Serialize(obj, JsonOptions);
   }
   
   public T? Deserialize<T>(string json)
   {
      return JsonSerializer.Deserialize<T>(json, JsonOptions);
   }
   
    private static JsonSerializerOptions CreateDefaultSerializeOptions()
    {
        var option = new JsonSerializerOptions()
        {
            PropertyNameCaseInsensitive = true,
            Converters = { new DictionaryStringToStringConverter() }
        };
        return option;
    }
}


public class DictionaryStringToStringConverter : JsonConverter<Dictionary<string, string>>
{
   public override bool CanConvert(Type typeToConvert)
   {
      if (typeToConvert == typeof(Dictionary<string, string>))
      {
         return true;
      }
      return false;
   }

   public override Dictionary<string, string> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
   {
      if (reader.TokenType != JsonTokenType.StartObject)
      {
         throw new JsonException($"JsonTokenType was of type {reader.TokenType}, only objects are supported");
      }

      var dictionary = new Dictionary<string, string>();
      while (reader.Read())
      {
         if (reader.TokenType == JsonTokenType.EndObject)
         {
            return dictionary;
         }
         if (reader.TokenType != JsonTokenType.PropertyName)
         {
            throw new JsonException("JsonTokenType was not PropertyName");
         }
         var propertyName = reader.GetString();
         if (string.IsNullOrWhiteSpace(propertyName))
         {
            throw new JsonException("Failed to get property name");
         }

         var value = ExtractValue(ref reader, options);
         dictionary.Add(propertyName!, $"{value}");
      }

      return dictionary;
      
   }

   public override void Write(Utf8JsonWriter writer, Dictionary<string, string> value, JsonSerializerOptions options)
   {
      writer.WriteStartObject();
      foreach (var item in value)
      {
         writer.WriteString(item.Key, item.Value);
      }
      writer.WriteEndObject();
   }
   
   private object? ExtractValue(ref Utf8JsonReader reader, JsonSerializerOptions options)
   {
      reader.Read();
      switch (reader.TokenType)
      {
         case JsonTokenType.String:
            if (reader.TryGetDateTime(out var date))
            {
               return date;
            }
            return reader.GetString();
         case JsonTokenType.False:
            return false;
         case JsonTokenType.True:
            return true;
         case JsonTokenType.Null:
            return null;
         case JsonTokenType.Number:
            if (reader.TryGetInt64(out var result))
            {
               return result;
            }
            return reader.GetDecimal();
         case JsonTokenType.StartObject:
            return Read(ref reader, null!, options);
         case JsonTokenType.StartArray:
            var list = new List<object?>();
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
               list.Add(ExtractValue(ref reader, options));
            }
            return list;
         default:
            throw new JsonException($"'{reader.TokenType}' is not supported");
      }
   }
}