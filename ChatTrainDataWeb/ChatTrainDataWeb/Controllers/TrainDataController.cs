﻿using ChatTrainDataWeb.Models.Contracts;
using ChatTrainDataWeb.Models.Repositories;
using Microsoft.AspNetCore.Mvc;

namespace ChatTrainDataWeb.Controllers;

[ApiController]
[Route("api/[controller]/[action]")]
public class TrainDataController : ControllerBase
{
    private IChatTrainDataRepo _chatTrainDataRepo;

    public TrainDataController(IChatTrainDataRepo chatTrainDataRepo)
    {
        _chatTrainDataRepo = chatTrainDataRepo;
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
}