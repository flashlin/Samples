namespace QueryKits.ExcelUtils;

public static class CsvExtension
{
    public static string ToCsvString(this IEnumerable<Dictionary<string, string>> dictList)
    {
        using var csvWriter = new CsvMemoryWriter();
        foreach (var item in dictList.Select((item,index)=> new {row = item, index }))
        {
            if (item.index == 0)
            {
                var headers = item.row.Keys.ToList();
                csvWriter.WriteHeaders(headers);
            }
            csvWriter.WriteRow(item.row);
        }
        return csvWriter.ToCsvString();
    }
}