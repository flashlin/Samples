﻿namespace ChatTrainDataWeb.Models.Repositories;

public class AddTrainDataDto
{
    public string Instruction { get; set; } = string.Empty;
    public string Input { get; set; } = string.Empty;
    public string Output { get; set; } = string.Empty;
}