using System.Text.Json;
using QueryKits.ExcelUtils;

namespace QueryKits.Services;

public class TextConverter
{
    public string ConvertText(string text)
    {
        var textFormat = GetTextFormat(text);
        if (textFormat == TextFormat.JsonArray)
        {
            return text.ToCsvString();
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
            return ConvertLineToMultipleLine(text);
        }

        var lines = text.Split(Environment.NewLine);
        return string.Join(",", lines);
    }

    private static string JsonToCsv(string text)
    {
        var jsonStr = $"[{text}]";
        return jsonStr.ToCsvString();
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

    private static string ConvertLineToMultipleLine(string text)
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
            return string.Join(newLine, text.Split('\t'));
        }
        return string.Join(newLine, text.Split(','));
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
            return TextFormat.Text;
        }

        if (TryDeserializeJsonArray(text)) return TextFormat.JsonArrayLine;
        if (TryDeserializeJson(text)) return TextFormat.JsonLine;
        return TextFormat.Line;
    }

    private static bool TryReadMultipleLine(string text)
    {
        var sr = new StringReader(text);
        sr.ReadLine();
        var line2 = sr.ReadLine();
        return line2 != null;
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