using ChatTrainDataWeb.Models.Entities;

namespace ChatTrainDataWeb.Models.Repositories;

public class ChatTrainDataRepo : IChatTrainDataRepo
{
    private readonly ChatTrainDataContext _dbContext;

    public ChatTrainDataRepo(ChatTrainDataContext dbContext)
    {
        _dbContext = dbContext;
    }

    public void AddData(TrainDataDto data)
    {
        _dbContext.TrainData.Add(new TrainDataEntity
        {
            Instruction = data.Instruction,
            Input = data.Input,
            Output = data.Output
        });
        _dbContext.SaveChanges();
    }

    public List<TrainDataEntity> GetDataPage(int startId, int pageSize)
    {
        return _dbContext.TrainData
            .Where(x => x.Id > startId)
            .Take(pageSize)
            .ToList();
    }
}