using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace T1.WebTools.CsvEx;

public class CsvSheet
{
   public static CsvSheet ReadFrom(string csvData)
   {
      using var strReader = new StringReader(csvData);
      var option = new CsvConfiguration(CultureInfo.InvariantCulture)
      {
         Delimiter = ",",
         Mode = CsvMode.RFC4180,
         Encoding = Encoding.UTF8,
         HasHeaderRecord = true,
      };
      using var csvReader = new CsvReader(strReader, option);
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