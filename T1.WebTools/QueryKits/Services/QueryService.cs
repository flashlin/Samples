using QueryKits.CsvEx;
using QueryKits.ExcelUtils;

namespace QueryKits.Services;

public class QueryService : IQueryService
{
    private readonly IReportRepo _reportRepo;

    public QueryService(IReportRepo reportRepo)
    {
        _reportRepo = reportRepo;
    }

    public List<ExcelSheet> QueryRawSql(string sql)
    {
        var result = new List<ExcelSheet>();
        var dataSets = _reportRepo.QueryMultipleRawSql(sql).ToList();
        foreach (var ds in dataSets)
        {
            if (ds.Rows.Count == 0)
            {
                continue;
            }
            var headers = ds.Rows[0].Keys
                .Select(name => new ExcelColumn
                {
                    Name = name
                }).ToList();
            var sheet = new ExcelSheet();
            sheet.Headers.AddRange(headers);
            foreach (var row in ds.Rows)
            {
                sheet.Rows.Add(row.ToDictionary(x=>x.Key, x=> $"{x.Value}"));
            }
            result.Add(sheet);
        }
        return result; 
    }
}