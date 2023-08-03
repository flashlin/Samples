using Microsoft.AspNetCore.Mvc;

namespace ChatTrainDataWeb.Controllers;

[ApiController]
[Route("api/[controller]/[action]")]
public class TrainDataController : ControllerBase
{
    [HttpPost]
    public GetDataPageResponse GetDataPage(GetDataPageRequest req)
    {
        return new GetDataPageResponse
        {
            Items = new List<TrainDataItem>
            {
                new TrainDataItem
                {
                    Id = 0,
                    Instruction = "translate SQL to JSON",
                    Input = "select id from customer",
                    Output = "{\"type\":\"select\",\"cols\":[\"id as id\"],\"fromCause\":\"customer as customer\"}",
                }
            }
        };
    }
}

public class TrainDataItem
{
    public int Id { get; set; }
    public string Instruction { get; set; } = string.Empty;
    public string Input { get; set; } = string.Empty;
    public string Output { get; set; } = string.Empty;
}

public class GetDataPageResponse
{
    public List<TrainDataItem> Items { get; set; } = new();
}

public class GetDataPageRequest
{
    public int Index { get; set; }
    public int PageSize { get; set; }
}