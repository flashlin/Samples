using System.Text.Json;
using QueryKits.CsvEx;
using QueryKits.ExcelUtils;
using T1.Standard.Collections.Generics;

namespace QueryKits.Services;

public class TextConverter
{
    public string ConvertText(string text)
    {
        var textFormat = GetTextFormat(text);
        if (textFormat == TextFormat.JsonArray)
        {
            return JsonHelper.ToCsvString(text);
        }

        if (textFormat == TextFormat.JsonArrayLine)
        {
            return ReSerializeJson(text);
        }

        if (textFormat == TextFormat.Json)
        {
            return JsonToCsv(text);
        }

        if (textFormat == TextFormat.Line)
        {
            return string.Join(Environment.NewLine, ConvertLineToMultipleLine(text));
        }

        var lines = text.Split(Environment.NewLine);
        return string.Join(",", lines);
    }

    public CsvSheet ConvertTextToCsvSheet(string text)
    {
        var textFormat = GetTextFormat(text);
        if (new[] {TextFormat.JsonArray, TextFormat.JsonArrayLine, TextFormat.Json}.Contains(textFormat))
        {
            var csv = JsonHelper.ToCsvString(text);
            return CsvSheet.ReadFromString(csv);
        }

        if (TextFormat.Csv == textFormat)
        {
            return CsvSheet.ReadFromString(text);
        }

        if (textFormat == TextFormat.Line)
        {
            var lines = ConvertLineToMultipleLine(text);
            return ConvertLinesToCsvSheet(lines);
        }

        return ConvertLinesToCsvSheet(ConvertTextToLines(text));
    }

    private static CsvSheet ConvertLinesToCsvSheet(IEnumerable<string> lines)
    {
        var csvSheet = new CsvSheet();
        var headerName = "Unknown";
        csvSheet.Headers.Add(new CsvHeader
        {
            ColumnType = ColumnType.String,
            Name = headerName
        });
        foreach (var line in lines)
        {
            csvSheet.Rows.Add(new Dictionary<string, string>()
            {
                {headerName, line}
            });
        }

        return csvSheet;
    }

    private static string JsonToCsv(string text)
    {
        var jsonStr = $"[{text}]";
        return JsonHelper.ToCsvString(jsonStr);
    }

    private static string ReSerializeJson(string text)
    {
        var objArray = JsonSerializer.Deserialize<List<dynamic>>(text);
        var jsonArrayStr = JsonSerializer.Serialize(objArray, new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        });
        return jsonArrayStr;
    }

    private static string[] ConvertLineToMultipleLine(string text)
    {
        var charGroup = text.GroupBy(x => x)
            .Select(x => new {x.Key, Count = x.Count()})
            .Where(x => new[] {'\t', ','}.Contains(x.Key))
            .ToDictionary(x => x.Key, x => x.Count);
        if (!charGroup.TryGetValue('\t', out var tabCount))
        {
            tabCount = 0;
        }

        if (!charGroup.TryGetValue(',', out var commaCount))
        {
            commaCount = 0;
        }

        var newLine = Environment.NewLine;
        if (tabCount > commaCount)
        {
            //return string.Join(newLine, text.Split('\t'));
            return text.Split('\t');
        }

        //return string.Join(newLine, text.Split(','));
        return text.Split(',');
    }

    public TextFormat GetTextFormat(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return TextFormat.Empty;
        }

        if (TryReadMultipleLine(text))
        {
            if (TryDeserializeJsonArray(text)) return TextFormat.JsonArray;
            if (TryDeserializeJson(text)) return TextFormat.Json;
            if (TryDeserializeCsv(text)) return TextFormat.Csv;
            return TextFormat.Text;
        }

        if (TryDeserializeJsonArray(text)) return TextFormat.JsonArrayLine;
        if (TryDeserializeJson(text)) return TextFormat.JsonLine;
        return TextFormat.Line;
    }

    private bool TryDeserializeCsv(string text)
    {
        var rc = CsvSheet.ParseCsvDelimiter(text);
        return rc.Success;
    }

    private static bool TryReadMultipleLine(string text)
    {
        var sr = new StringReader(text);
        sr.ReadLine();
        var line2 = sr.ReadLine();
        return line2 != null;
    }

    private IEnumerable<string> ConvertTextToLines(string text)
    {
        var sr = new StringReader(text);
        do
        {
            var line = sr.ReadLine();
            if (line == null)
            {
                break;
            }

            yield return line;
        } while (true);
    }

    private static bool TryDeserializeJson(string text)
    {
        try
        {
            var obj = JsonSerializer.Deserialize<dynamic>(text);
            if (obj is JsonElement jsonElement)
            {
                return jsonElement.ValueKind == JsonValueKind.Object;
            }

            return obj != null;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryDeserializeJsonArray(string text)
    {
        try
        {
            var obj = JsonSerializer.Deserialize<List<dynamic>>(text);
            return obj != null;
        }
        catch
        {
            return false;
        }
    }
}