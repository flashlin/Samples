namespace ChatTrainDataWeb.Models.Repositories;

public class UpdateTrainDataDto
{
    public int Id { get; set; }
    public string Instruction { get; set; } = string.Empty;
    public string Input { get; set; } = string.Empty;
    public string Output { get; set; } = string.Empty;
}