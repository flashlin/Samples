namespace ChatTrainDataWeb.Models.Contracts;

public class TrainDataItem
{
    public int Id { get; set; }
    public string Instruction { get; set; } = string.Empty;
    public string Input { get; set; } = string.Empty;
    public string Output { get; set; } = string.Empty;
}