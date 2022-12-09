using Hangfire;

namespace HangfireWebDemo.Controllers;

public class MgmtController
{
    private IServiceProvider _serviceProvider;

    public MgmtController(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }
    public interface IBaseRecurringJob
    {
        string JobName { get; }
        string Cron { get; }
        void Process();
    } 
    
    public void GetJob(string jobName)
    {
        var recurringJobTypes = typeof(MgmtController).Assembly.GetTypes()
            .Where(x => x.BaseType == typeof(IBaseRecurringJob))
            .ToList();

        foreach (var recurringJobType in recurringJobTypes)
        {
            var recurringJobInstance = (IBaseRecurringJob)_serviceProvider.GetService(recurringJobType)!;
            if (recurringJobInstance.JobName == jobName)
            {
                RecurringJob.AddOrUpdate(recurringJobInstance.JobName,
                    () => recurringJobInstance.Process(), recurringJobInstance.Cron);
            }
        } 
    }
}