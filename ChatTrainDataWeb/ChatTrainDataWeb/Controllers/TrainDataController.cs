using System.Globalization;
using System.Text;
using ChatTrainDataWeb.Models.Contracts;
using ChatTrainDataWeb.Models.Entities;
using ChatTrainDataWeb.Models.Repositories;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.AspNetCore.Mvc;

namespace ChatTrainDataWeb.Controllers;

[ApiController]
[Route("api/[controller]/[action]")]
public class TrainDataController : ControllerBase
{
    private readonly IChatTrainDataRepo _chatTrainDataRepo;

    public TrainDataController(IChatTrainDataRepo chatTrainDataRepo)
    {
        _chatTrainDataRepo = chatTrainDataRepo;
    }

    [HttpGet]
    public IActionResult Index()
    {
        return Content("OK");
    }
    
    [HttpPost]
    public GetDataPageResponse GetDataPage(GetDataPageRequest req)
    {
        return new GetDataPageResponse
        {
            Items = _chatTrainDataRepo.GetDataPage(req.StartIndex, req.PageSize)
                .Select(x => new TrainDataItem
                {
                    Id = x.Id,
                    Instruction = x.Instruction,
                    Input = x.Input,
                    Output = x.Output, 
                }).ToList()
        };
    }

    [HttpPost]
    public void AddData(AddDataRequest req)
    {
        _chatTrainDataRepo.AddData(new AddTrainDataDto
        {
            Instruction = req.Instruction,
            Input = req.Input,
            Output = req.Output,
        });
    }
    
    [HttpPost]
    public void UpdateData(UpdateDataRequest req)
    {
        _chatTrainDataRepo.UpdateData(new UpdateTrainDataDto
        {
            Id = req.Id,
            Instruction = req.Instruction,
            Input = req.Input,
            Output = req.Output,
        });
    }

    [HttpGet]
    public IActionResult ExportToT5Csv()
    {
        var csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture);
        using var memoryStream = new MemoryStream();
        using var writer = new StreamWriter(memoryStream);
        using var csv = new CsvWriter(writer, csvConfig);
        csv.WriteHeader<TrainDataEntity>();
        csv.NextRecord();
        _chatTrainDataRepo.Fetch(x =>
        {
            csv.WriteRecord(x);
            csv.NextRecord();
        });
        csv.Flush();
        var bytes = memoryStream.ToArray();
        return File(bytes, "text/csv", "users.csv");
    }
}