using QueryKits.CsvEx;

namespace QueryKits.Services;

public class QueryService : IQueryService
{
    private readonly IReportRepo _reportRepo;

    public QueryService(IReportRepo reportRepo)
    {
        _reportRepo = reportRepo;
    }

    public List<CsvSheet> QueryRawSql(string sql)
    {
        var result = new List<CsvSheet>();
        var dataSets = _reportRepo.QueryMultipleRawSql(sql).ToList();
        foreach (var ds in dataSets)
        {
            if (ds.Rows.Count == 0)
            {
                continue;
            }
            var headers = ds.Rows[0].Keys
                .Select(name => new CsvHeader
                {
                    ColumnType = ColumnType.String,
                    Name = name
                }).ToList();
            var sheet = new CsvSheet();
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