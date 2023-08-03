using ChatTrainDataWeb.Models.Constracts;
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

    public void AddData(AddDataRequest req)
    {
        
    }
}

public class AddDataRequest
{
}