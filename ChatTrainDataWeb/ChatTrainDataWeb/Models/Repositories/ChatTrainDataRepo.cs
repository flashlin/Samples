using ChatTrainDataWeb.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace ChatTrainDataWeb.Models.Repositories;

public class ChatTrainDataRepo : IChatTrainDataRepo
{
    private readonly ChatTrainDataContext _dbContext;

    public ChatTrainDataRepo(ChatTrainDataContext dbContext)
    {
        _dbContext = dbContext;
    }

    public void AddData(AddTrainDataDto data)
    {
        _dbContext.TrainData.Add(new TrainDataEntity
        {
            Instruction = data.Instruction,
            Input = data.Input,
            Output = data.Output
        });
        _dbContext.SaveChanges();
    }
    
    public void UpdateData(UpdateTrainDataDto data)
    {
        var existingEntity = new TrainDataEntity { Id = data.Id };
        _dbContext.Entry(existingEntity).CurrentValues.SetValues(new 
        {
            Instruction = data.Instruction,
            Input = data.Input,
            Output = data.Output
        });
        _dbContext.Entry(existingEntity).State = EntityState.Modified;
        _dbContext.SaveChanges();
    }

    public void Fetch(Action<TrainDataEntity> action)
    {
        foreach (var trainDataEntity in _dbContext.TrainData)
        {
            action(trainDataEntity);
        }
    }

    public List<TrainDataEntity> GetDataPage(int startId, int pageSize)
    {
        return _dbContext.TrainData
            .Where(x => x.Id > startId)
            .Take(pageSize)
            .ToList();
    }
}