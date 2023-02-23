using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace T1.WebTools.CsvEx;

public class CsvSheet
{
   
   public static string ParseHeaderDelimiter(string line)
   {
      if (line.Contains('\t'))
      {
         return "\t";
      }
      if (line.Contains(','))
      {
         return ",";
      }
      return " ";
   }

   public static string ParseHeaderDelimiterFromFile(string csvFile)
   {
      using var stream = new FileStream(csvFile, FileMode.Open);
      using var reader = new StreamReader(stream, Encoding.UTF8);
      var line = reader.ReadLine()!;
      return ParseHeaderDelimiter(line);
   }

   public static CsvSheet ReadFromStream(Stream csvStream, string delimiter)
   {
      using var textReader = new StreamReader(csvStream);
      return ReadFromTextReader(textReader, delimiter);
   }

   public static CsvSheet ReadFromString(string csvData)
   {
      using var strReader = new StringReader(csvData);
      return ReadFromTextReader(strReader, ",");
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

   public List<CsvHeader> Headers { get; init; } = new();
   public List<Dictionary<string, string>> Rows = new();

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
}

public class CsvDataProcessor
{
   public void CsvToJson()
   {
      
   }
   
   public void Merge(CsvSheet master, List<CsvSheet> slaves)
   {
      foreach (var row in master.Rows)
      {
      }
   }
}