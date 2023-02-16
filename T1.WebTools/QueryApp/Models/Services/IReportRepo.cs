namespace QueryApp.Models.Services;

public interface IReportRepo
{
    List<string> GetAllTableNames();
}